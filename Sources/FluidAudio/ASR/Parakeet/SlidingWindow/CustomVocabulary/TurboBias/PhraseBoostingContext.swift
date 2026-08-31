// The context graph and variative-BPE construction are derived from NVIDIA NeMo Speech,
// Copyright NVIDIA Corporation, licensed under Apache License 2.0.
import Foundation

/// Decode-time phrase boosting parameters from NVIDIA NeMo's TurboBias implementation.
///
/// The effective token score is `acoustic + alpha * graphTransition`. NeMo's published
/// defaults are a context score of 1, depth scaling of 2, and an unknown-token score of 0.
public struct PhraseBoostingConfig: Sendable, Equatable {
    public let contextScore: Float
    public let depthScaling: Float
    public let alpha: Float
    public let unknownScore: Float
    /// Use TurboBias 2.0's variative-BPE graph so casing and BPE segmentation
    /// differences reach the same phrase state without duplicating phrase strings.
    public let caseInsensitive: Bool
    /// Controls where a greedy BPE token's reward lands across its character states.
    /// NVIDIA's conservative greedy-decoding default is 10.
    public let variativeScoringTemperature: Float
    /// Prevent character-by-character paths from receiving an earlier/larger reward
    /// than the original greedy BPE path.
    public let penalizeSubsplits: Bool

    public init(
        contextScore: Float = 1,
        depthScaling: Float = 2,
        alpha: Float = 1,
        unknownScore: Float = 0,
        caseInsensitive: Bool = true,
        variativeScoringTemperature: Float = 10,
        penalizeSubsplits: Bool = true
    ) {
        self.contextScore = contextScore
        self.depthScaling = depthScaling
        self.alpha = alpha
        self.unknownScore = unknownScore
        self.caseInsensitive = caseInsensitive
        self.variativeScoringTemperature = variativeScoringTemperature
        self.penalizeSubsplits = penalizeSubsplits
    }
}

public enum PhraseBoostingError: Error, LocalizedError, Equatable {
    case unsupportedModel
    case fullVocabularyJointUnavailable
    case invalidConfiguration
    case emptyPhrase
    case untokenizablePhrase(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel:
            return "Decode-time phrase boosting is not available for this Parakeet model."
        case .fullVocabularyJointUnavailable:
            return "The full-vocabulary Parakeet joint model is not installed."
        case .invalidConfiguration:
            return "Phrase boosting scores must be finite, nonnegative, and use depth scaling of at least 1."
        case .emptyPhrase:
            return "Dictionary phrases must contain text."
        case .untokenizablePhrase(let phrase):
            return "The Parakeet tokenizer could not represent ‘\(phrase)’ without an unknown token."
        }
    }
}

/// Immutable weighted Aho-Corasick graph used by NeMo's TurboBias greedy decoder.
///
/// This is deliberately a value prepared before decoding. A decoder carries only one integer
/// state while the graph is shared by every token step and worker.
public struct PhraseBoostingContext: Sendable {
    struct Selection: Sendable {
        let token: Int
        let nextState: Int
    }

    private struct Transition: Sendable {
        let nextState: Int
        let score: Float
    }

    fileprivate struct TokenWithLength: Sendable {
        let token: Int
        let length: Int
    }

    fileprivate struct VariativeRepresentation: Sendable {
        let canonicalLengths: [Int]
        let tokenGroups: [[TokenWithLength]]
    }

    private struct Node: Sendable {
        var children: [Int: Int] = [:]
        /// Character-level backbone used to calculate Aho-Corasick failure links.
        /// Variative and merged-token arcs live only in `children`.
        var primaryChildren: [Int: Int] = [:]
        var fail = 0
        var tokenScore: Float = 0
        var nodeScore: Float = 0
        var isEnd = false
        var phraseIndices: [Int] = []
        var outputPhraseIndices: [Int] = []
        var formattingPhraseIndices: [Int] = []
    }

    public let phrases: [String]
    public let tokenizedPhrases: [[Int]]
    public let config: PhraseBoostingConfig

    let blankID: Int
    private let nodes: [Node]
    private let phraseEndBoundaryTokens: Set<Int>
    private let maximumRootTransitionScore: Float

    var rootState: Int { 0 }

    /// Return the base-token transition when fusion cannot possibly reselect.
    ///
    /// With alpha zero, fusion is disabled. At the root, the full-vocabulary
    /// joint is also unnecessary when the primary token already receives the
    /// graph's largest root reward. In both cases the graph state can advance
    /// without running the optional joint model.
    func selectionWithoutAcousticScores(
        baseToken: Int,
        baseProbability: Float? = nil,
        state: Int
    ) -> Selection? {
        guard baseToken >= 0, baseToken < blankID else { return nil }
        let fusionDisabled = config.alpha == 0
        let baseTransition = transition(from: state, token: baseToken)
        let baseAlreadyHasMaximumReward =
            state == rootState
            && baseTransition.score >= maximumRootTransitionScore - 0.000_001
        let probabilityProvesBaseCannotLose: Bool
        if state == rootState,
            let baseProbability,
            baseProbability.isFinite,
            baseProbability > 0,
            baseProbability <= 1
        {
            // Every competing probability is at most `1 - p(base)`. Therefore
            // `log((1-p)/p)` bounds its acoustic-logit advantage. If even that
            // bound plus the graph's largest possible reward advantage cannot
            // beat the base token, the full-vocabulary joint cannot change the
            // result. A small logit margin covers independent Core ML exports.
            let rewardAdvantage = max(
                0,
                config.alpha * (maximumRootTransitionScore - baseTransition.score)
            )
            let conservativeThreshold = Float(
                1 / (1 + Foundation.exp(-Double(rewardAdvantage + 0.01)))
            )
            probabilityProvesBaseCannotLose = baseProbability >= conservativeThreshold
        } else {
            probabilityProvesBaseCannotLose = false
        }
        guard fusionDisabled || baseAlreadyHasMaximumReward || probabilityProvesBaseCannotLose
        else { return nil }
        return Selection(token: baseToken, nextState: baseTransition.nextState)
    }

    init(
        phrases: [String],
        vocabulary: [Int: String],
        blankID: Int,
        config: PhraseBoostingConfig
    ) throws {
        guard config.contextScore.isFinite, config.contextScore >= 0,
            config.depthScaling.isFinite, config.depthScaling >= 1,
            config.alpha.isFinite, config.alpha >= 0,
            config.unknownScore.isFinite, config.unknownScore >= 0,
            config.variativeScoringTemperature.isFinite,
            config.variativeScoringTemperature >= 0
        else {
            throw PhraseBoostingError.invalidConfiguration
        }

        let normalizedPhrases = phrases.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard normalizedPhrases.allSatisfy({ !$0.isEmpty }) else {
            throw PhraseBoostingError.emptyPhrase
        }

        let tokenizer = ParakeetSentencePieceBPETokenizer(
            vocabulary: vocabulary,
            blankID: blankID
        )
        let tokenized = try normalizedPhrases.map { phrase in
            guard let tokens = tokenizer.encode(phrase), !tokens.isEmpty else {
                throw PhraseBoostingError.untokenizablePhrase(phrase)
            }
            return tokens
        }

        var buildingNodes = [Node()]
        for (phraseIndex, phrase) in normalizedPhrases.enumerated() {
            let terminalState: Int
            if config.caseInsensitive {
                guard let representation = tokenizer.variativeRepresentation(for: phrase.lowercased())
                else {
                    throw PhraseBoostingError.untokenizablePhrase(phrase)
                }
                terminalState = Self.addVariativePhrase(
                    representation,
                    config: config,
                    to: &buildingNodes
                )
            } else {
                terminalState = Self.addGreedyPhrase(
                    tokenized[phraseIndex],
                    config: config,
                    to: &buildingNodes
                )
            }

            buildingNodes[terminalState].isEnd = true
            if !buildingNodes[terminalState].phraseIndices.contains(phraseIndex) {
                buildingNodes[terminalState].phraseIndices.append(phraseIndex)
            }
            if config.caseInsensitive,
                phrase.lowercased() != phrase,
                !buildingNodes[terminalState].formattingPhraseIndices.contains(phraseIndex)
            {
                buildingNodes[terminalState].formattingPhraseIndices.append(phraseIndex)
            }
        }

        for index in buildingNodes.indices {
            buildingNodes[index].outputPhraseIndices = buildingNodes[index].phraseIndices
        }

        Self.fillFailureLinks(in: &buildingNodes)

        self.phrases = normalizedPhrases
        self.tokenizedPhrases = tokenized
        self.config = config
        self.blankID = blankID
        self.nodes = buildingNodes
        self.maximumRootTransitionScore = max(
            config.unknownScore,
            buildingNodes[0].children.values.map { buildingNodes[$0].nodeScore }.max() ?? 0
        )
        self.phraseEndBoundaryTokens = Set(
            vocabulary.compactMap { token, piece in
                if piece.hasPrefix(ASRConstants.sentencePieceWordBoundary) { return token }
                guard let scalar = piece.unicodeScalars.first else { return token }
                return CharacterSet.alphanumerics.contains(scalar) ? nil : token
            })
    }

    private static func addGreedyPhrase(
        _ tokens: [Int],
        config: PhraseBoostingConfig,
        to nodes: inout [Node]
    ) -> Int {
        var state = 0
        for (depth, token) in tokens.enumerated() {
            let tokenScore = score(depth: depth, config: config)
            if let existing = nodes[state].children[token] {
                let sharedScore = max(tokenScore, nodes[existing].tokenScore)
                nodes[existing].tokenScore = sharedScore
                nodes[existing].nodeScore = nodes[state].nodeScore + sharedScore
                nodes[state].primaryChildren[token] = existing
                state = existing
            } else {
                let next = nodes.count
                nodes.append(
                    Node(
                        tokenScore: tokenScore,
                        nodeScore: nodes[state].nodeScore + tokenScore
                    )
                )
                nodes[state].children[token] = next
                nodes[state].primaryChildren[token] = next
                state = next
            }
        }
        return state
    }

    /// Port of NeMo `ContextGraph.build_from_var_bpe` used by TurboBias 2.0.
    /// Character states are the scoring backbone; case and merged-token arcs
    /// converge on those states, so every valid BPE segmentation earns the same
    /// total phrase reward.
    private static func addVariativePhrase(
        _ representation: VariativeRepresentation,
        config: PhraseBoostingConfig,
        to nodes: inout [Node]
    ) -> Int {
        let tokenCount = representation.tokenGroups.count
        var tokenScores = [Float](repeating: 0, count: tokenCount)
        var isPrimaryEndpoint = [Bool](repeating: false, count: tokenCount)
        var primaryScores = [Float](repeating: 0, count: tokenCount)
        var primaryBackJumps = [Int](repeating: 0, count: tokenCount)

        var offset = 0
        for (depth, canonicalLength) in representation.canonicalLengths.enumerated() {
            let endpoint = offset + canonicalLength - 1
            isPrimaryEndpoint[endpoint] = true
            let primaryScore = score(depth: depth, config: config)
            let weights = softmaxWeights(
                count: canonicalLength,
                temperature: config.variativeScoringTemperature
            )
            for index in 0..<canonicalLength {
                tokenScores[offset + index] = primaryScore * weights[index]
            }
            primaryScores[endpoint] = primaryScore
            primaryBackJumps[endpoint] = canonicalLength
            offset += canonicalLength
        }

        var state = 0
        var statesByCanonicalPosition = [0]
        var accumulatedScore: Float = 0

        for index in representation.tokenGroups.indices {
            let group = representation.tokenGroups[index]
            precondition(!group.isEmpty)
            let primaryToken = group[0].token
            accumulatedScore += tokenScores[index]
            let nextState: Int

            if let existing = nodes[state].children[primaryToken] {
                nextState = existing
                nodes[state].primaryChildren[primaryToken] = existing
                if isPrimaryEndpoint[index] {
                    let sourceIndex = statesByCanonicalPosition.count - primaryBackJumps[index]
                    let primaryPotential =
                        nodes[statesByCanonicalPosition[sourceIndex]].nodeScore + primaryScores[index]
                    nodes[existing].nodeScore = max(nodes[existing].nodeScore, primaryPotential)
                }
            } else {
                let potential: Float
                if config.penalizeSubsplits, !isPrimaryEndpoint[index] {
                    potential = max(0, accumulatedScore - nodes[state].nodeScore)
                } else {
                    potential = accumulatedScore
                }
                nextState = nodes.count
                nodes.append(Node(nodeScore: potential))
                nodes[state].children[primaryToken] = nextState
                nodes[state].primaryChildren[primaryToken] = nextState
            }

            for alternative in group.dropFirst() {
                if alternative.length == 1 {
                    nodes[state].children[alternative.token] = nextState
                } else {
                    let sourceIndex = statesByCanonicalPosition.count - alternative.length
                    nodes[statesByCanonicalPosition[sourceIndex]].children[alternative.token] = nextState
                }
            }

            statesByCanonicalPosition.append(nextState)
            state = nextState
        }
        return state
    }

    private static func fillFailureLinks(in nodes: inout [Node]) {
        var queue = Array(nodes[0].primaryChildren.values)
        var queueIndex = 0
        var visited: Set<Int> = [0]
        for child in queue {
            nodes[child].fail = 0
        }

        while queueIndex < queue.count {
            let current = queue[queueIndex]
            queueIndex += 1
            guard visited.insert(current).inserted else { continue }

            for (token, child) in nodes[current].primaryChildren where !visited.contains(child) {
                var failure = nodes[current].fail
                while failure != 0 && nodes[failure].primaryChildren[token] == nil {
                    failure = nodes[failure].fail
                }
                if let suffix = nodes[failure].primaryChildren[token], suffix != child {
                    nodes[child].fail = suffix
                } else {
                    nodes[child].fail = 0
                }
                for phraseIndex in nodes[nodes[child].fail].outputPhraseIndices
                where !nodes[child].outputPhraseIndices.contains(phraseIndex) {
                    nodes[child].outputPhraseIndices.append(phraseIndex)
                }
                queue.append(child)
            }
        }
    }

    private static func score(depth: Int, config: PhraseBoostingConfig) -> Float {
        guard depth > 0 else { return config.contextScore }
        return config.contextScore * config.depthScaling
            + Float(Foundation.log(Double(depth + 1)))
    }

    private static func softmaxWeights(count: Int, temperature: Float) -> [Float] {
        guard count > 1 else { return [1] }
        let logits = (0..<count).map {
            Foundation.pow(Double($0 + 1), Double(temperature))
        }
        let maximum = logits.max() ?? 0
        let exponentials = logits.map { Foundation.exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        return exponentials.map { Float($0 / denominator) }
    }

    /// Re-select a non-blank greedy token after shallow fusion with the phrase graph.
    ///
    /// NeMo first decides blank versus non-blank from the unmodified acoustic model. If the
    /// result is non-blank, fusion is allowed to choose another non-blank vocabulary token.
    /// The caller enforces that blank-category decision before invoking this method.
    func select(baseToken: Int, acousticScores: [Float], state: Int) -> Selection {
        guard baseToken >= 0, baseToken < blankID, baseToken < acousticScores.count else {
            return Selection(token: baseToken, nextState: state)
        }

        let baseTransition = transition(from: state, token: baseToken)
        guard config.alpha > 0 else {
            return Selection(token: baseToken, nextState: baseTransition.nextState)
        }
        var bestToken = baseToken
        var bestTransition = baseTransition
        var bestScore = acousticScores[baseToken] + config.alpha * baseTransition.score

        let candidateCount = min(blankID, acousticScores.count)
        for token in 0..<candidateCount where token != baseToken {
            let candidateTransition = transition(from: state, token: token)
            // The optimized JointDecision model remains authoritative for the
            // unboosted token. RNNTJoint is a separately compiled view of the same
            // weights, so only a strictly better graph transition may override it;
            // numerical differences between Core ML exports cannot change unrelated
            // words when a dictionary is enabled.
            guard candidateTransition.score > baseTransition.score else { continue }
            let candidateScore = acousticScores[token] + config.alpha * candidateTransition.score
            if candidateScore > bestScore {
                bestToken = token
                bestTransition = candidateTransition
                bestScore = candidateScore
            }
        }
        return Selection(token: bestToken, nextState: bestTransition.nextState)
    }

    func matchingPhrases(in tokens: [Int]) -> [String] {
        var matched = Set<Int>()
        var state = rootState
        for (index, token) in tokens.enumerated() {
            state = transition(from: state, token: token).nextState
            guard isPhraseEndBoundary(at: index + 1, tokens: tokens) else { continue }
            for phraseIndex in nodes[state].outputPhraseIndices {
                matched.insert(phraseIndex)
            }
        }
        return matched.sorted().map { phrases[$0] }
    }

    /// Preserve the user's mixed-case or acronym spelling when decoding followed
    /// one of the convenience capitalization paths.
    ///
    /// Lowercase entries are left to the model so ordinary sentence-start casing
    /// remains natural. This mirrors the existing CTC vocabulary path's promise
    /// that deliberately styled product names remain exact.
    func formattedText(in tokens: [Int], vocabulary: [Int: String]) -> String {
        guard !tokens.isEmpty else { return "" }

        var pieces: [String] = []
        var index = 0
        while index < tokens.count {
            if let match = formattingMatch(startingAt: index, tokens: tokens) {
                let canonical = phrases[match.phraseIndex]
                    .split(whereSeparator: { $0.isWhitespace })
                    .joined(separator: ASRConstants.sentencePieceWordBoundary)
                pieces.append(ASRConstants.sentencePieceWordBoundary + canonical)
                index += match.length
            } else {
                if let piece = vocabulary[tokens[index]], !piece.isEmpty {
                    pieces.append(piece)
                }
                index += 1
            }
        }

        return pieces.joined()
            .replacingOccurrences(of: ASRConstants.sentencePieceWordBoundary, with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    private func formattingMatch(
        startingAt startIndex: Int,
        tokens: [Int]
    ) -> (phraseIndex: Int, length: Int)? {
        guard config.alpha > 0 else { return nil }
        var state = rootState
        var index = startIndex
        var best: (phraseIndex: Int, length: Int)?

        while index < tokens.count, let child = nodes[state].children[tokens[index]] {
            state = child
            index += 1
            guard isPhraseEndBoundary(at: index, tokens: tokens) else { continue }
            for phraseIndex in nodes[state].formattingPhraseIndices {
                let length = index - startIndex
                if let current = best {
                    if length > current.length
                        || (length == current.length && phraseIndex < current.phraseIndex)
                    {
                        best = (phraseIndex, length)
                    }
                } else {
                    best = (phraseIndex, length)
                }
            }
        }
        return best
    }

    private func isPhraseEndBoundary(
        at index: Int,
        tokens: [Int]
    ) -> Bool {
        guard index < tokens.count else { return true }
        return phraseEndBoundaryTokens.contains(tokens[index])
    }

    private func transition(from originalState: Int, token: Int) -> Transition {
        var state = nodes.indices.contains(originalState) ? originalState : rootState
        var score: Float = 0

        while state != rootState && nodes[state].children[token] == nil {
            let failure = nodes[state].fail
            // NeMo deliberately does not remove a completed phrase's reward on backoff.
            if !nodes[state].isEnd {
                score += nodes[failure].nodeScore - nodes[state].nodeScore
            }
            state = failure
        }

        guard let next = nodes[state].children[token] else {
            return Transition(nextState: rootState, score: score + config.unknownScore)
        }
        score += nodes[next].nodeScore - nodes[state].nodeScore
        return Transition(nextState: next, score: score)
    }
}

/// SentencePiece BPE encoding reconstructed from the ordered Parakeet vocabulary.
///
/// Parakeet v2's piece ID is its BPE merge rank (the bundled NeMo `.vocab` score is
/// `-(id - 1)`). This produces the same token IDs without bundling a second tokenizer asset.
private struct ParakeetSentencePieceBPETokenizer {
    private let tokenToID: [String: Int]
    private let canonicalIDByTokenID: [Int: Int]
    private let alternativesByCanonicalID: [Int: [Int]]
    private let canonicalSplitByTokenID: [Int: [Int]]
    private let tokenIDsByCanonicalSplit: [[Int]: [Int]]
    private let maximumTokenLength: Int

    init(vocabulary: [Int: String], blankID: Int) {
        var byPiece: [String: Int] = [:]
        for (id, piece) in vocabulary where id >= 0 && id < blankID && piece != "<unk>" {
            byPiece[piece] = min(id, byPiece[piece] ?? id)
        }
        tokenToID = byPiece

        var canonicalIDs: [Int: Int] = [:]
        for (piece, id) in byPiece {
            let lowercase = piece.lowercased()
            canonicalIDs[id] = lowercase != piece ? (byPiece[lowercase] ?? id) : id
        }
        canonicalIDByTokenID = canonicalIDs

        var alternatives: [Int: [Int]] = [:]
        for id in canonicalIDs.keys.sorted() {
            guard let canonicalID = canonicalIDs[id] else { continue }
            alternatives[canonicalID, default: []].append(id)
        }
        for canonicalID in Array(alternatives.keys) {
            alternatives[canonicalID]?.sort { left, right in
                if left == canonicalID { return true }
                if right == canonicalID { return false }
                return left < right
            }
        }
        alternativesByCanonicalID = alternatives

        var splits: [Int: [Int]] = [:]
        var bySplit: [[Int]: [Int]] = [:]
        var maxLength = 1
        for (piece, id) in byPiece {
            let canonicalID = canonicalIDs[id] ?? id
            let split: [Int]
            if piece.count == 1 || (piece.hasPrefix("<") && piece.hasSuffix(">")) {
                split = [canonicalID]
            } else {
                let candidate = piece.compactMap { character -> Int? in
                    guard let characterID = byPiece[String(character)] else { return nil }
                    return canonicalIDs[characterID] ?? characterID
                }
                split = candidate.count == piece.count ? candidate : [canonicalID]
            }
            splits[id] = split
            bySplit[split, default: []].append(id)
            maxLength = max(maxLength, split.count)
        }
        for split in Array(bySplit.keys) {
            bySplit[split]?.sort()
        }
        canonicalSplitByTokenID = splits
        tokenIDsByCanonicalSplit = bySplit
        maximumTokenLength = maxLength
    }

    func encode(_ text: String) -> [Int]? {
        let compatibilityNormalized = text.precomposedStringWithCompatibilityMapping
        let words = compatibilityNormalized.split(whereSeparator: { $0.isWhitespace })
        guard !words.isEmpty else { return nil }
        let sentencePieceText = "▁" + words.map(String.init).joined(separator: "▁")
        var pieces = sentencePieceText.map(String.init)

        while pieces.count > 1 {
            var bestIndex: Int?
            var bestID = Int.max
            for index in 0..<(pieces.count - 1) {
                guard let id = tokenToID[pieces[index] + pieces[index + 1]] else { continue }
                if id < bestID {
                    bestID = id
                    bestIndex = index
                }
            }
            guard let bestIndex else { break }
            pieces[bestIndex] += pieces[bestIndex + 1]
            pieces.remove(at: bestIndex + 1)
        }

        let ids = pieces.compactMap { tokenToID[$0] }
        return ids.count == pieces.count ? ids : nil
    }

    /// NeMo TurboBias 2.0 variative-BPE representation. The phrase's greedy
    /// BPE tokens define score boundaries, while the graph accepts every vocab
    /// token whose lowercased character decomposition covers the same span.
    func variativeRepresentation(
        for text: String
    ) -> PhraseBoostingContext.VariativeRepresentation? {
        guard let greedyIDs = encode(text), !greedyIDs.isEmpty else { return nil }
        let canonicalLengths = greedyIDs.compactMap { canonicalSplitByTokenID[$0]?.count }
        guard canonicalLengths.count == greedyIDs.count else { return nil }
        let canonicalIDs = greedyIDs.flatMap { canonicalSplitByTokenID[$0] ?? [] }
        guard canonicalIDs.count == canonicalLengths.reduce(0, +) else { return nil }

        var groups = [[PhraseBoostingContext.TokenWithLength]]()
        groups.reserveCapacity(canonicalIDs.count)
        for index in canonicalIDs.indices {
            let canonicalID = canonicalIDs[index]
            var group = (alternativesByCanonicalID[canonicalID] ?? [canonicalID]).map {
                PhraseBoostingContext.TokenWithLength(token: $0, length: 1)
            }

            let earliestStart = max(0, index - maximumTokenLength)
            if earliestStart < index {
                for start in earliestStart..<index {
                    let split = Array(canonicalIDs[start...index])
                    for token in tokenIDsByCanonicalSplit[split] ?? []
                    where !group.contains(where: { $0.token == token }) {
                        group.append(
                            PhraseBoostingContext.TokenWithLength(
                                token: token,
                                length: index - start + 1
                            )
                        )
                    }
                }
            }
            guard !group.isEmpty else { return nil }
            groups.append(group)
        }
        return PhraseBoostingContext.VariativeRepresentation(
            canonicalLengths: canonicalLengths,
            tokenGroups: groups
        )
    }
}
