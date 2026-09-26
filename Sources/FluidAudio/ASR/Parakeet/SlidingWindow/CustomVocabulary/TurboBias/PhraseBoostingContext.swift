// The context graph and variative-BPE construction are derived from NVIDIA NeMo Speech,
// Copyright NVIDIA Corporation, licensed under Apache License 2.0.
import ContextBiasing
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
        case .untokenizablePhrase:
            return "The Parakeet tokenizer could not represent one dictionary phrase without an unknown token."
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
        /// Acoustic-model probability for a replacement token. The optimized
        /// joint's probability remains authoritative when TurboBias keeps its
        /// original token.
        let replacementProbability: Float?
    }

    private typealias Transition = PhraseBoostingGraph.Transition
    fileprivate typealias TokenWithLength = VariativeBPERepresentation.TokenWithLength
    fileprivate typealias VariativeRepresentation = VariativeBPERepresentation

    public let phrases: [String]
    public let tokenizedPhrases: [[Int]]
    public let config: PhraseBoostingConfig
    public let skippedPhraseCount: Int

    let blankID: Int
    private let graph: PhraseBoostingGraph
    private let formattingPhraseIndices: Set<Int>
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
        return Selection(
            token: baseToken,
            nextState: baseTransition.nextState,
            replacementProbability: nil
        )
    }

    init(
        phrases: [String],
        vocabulary: [Int: String],
        blankID: Int,
        config: PhraseBoostingConfig,
        phraseWeights: [Float]? = nil,
        skipUnsupportedPhrases: Bool = false
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
        if let phraseWeights {
            guard phraseWeights.count == phrases.count,
                phraseWeights.allSatisfy({ $0.isFinite && $0 > 0 })
            else { throw PhraseBoostingError.invalidConfiguration }
        }

        let normalizedPhrases = phrases.map {
            $0.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !normalizedPhrases.isEmpty,
            normalizedPhrases.allSatisfy({ !$0.isEmpty })
        else {
            throw PhraseBoostingError.emptyPhrase
        }

        let tokenizer = ParakeetSentencePieceBPETokenizer(
            vocabulary: vocabulary,
            blankID: blankID
        )
        var acceptedPhrases: [String] = []
        var tokenizedPhrases: [[Int]] = []
        var variativeRepresentations: [VariativeRepresentation?] = []
        var acceptedWeights: [Float] = []
        var skippedPhrases = 0
        var firstUnsupportedPhrase: String?
        for (phraseIndex, phrase) in normalizedPhrases.enumerated() {
            let tokens: [Int]?
            let representation: VariativeRepresentation?
            if config.caseInsensitive {
                let lowercasePhrase = phrase.lowercased()
                tokens = tokenizer.encode(phrase) ?? tokenizer.encode(lowercasePhrase)
                representation = tokenizer.variativeRepresentation(for: lowercasePhrase)
            } else {
                tokens = tokenizer.encode(phrase)
                representation = nil
            }
            guard let tokens, !tokens.isEmpty,
                !config.caseInsensitive || representation != nil
            else {
                guard skipUnsupportedPhrases else {
                    throw PhraseBoostingError.untokenizablePhrase(phrase)
                }
                skippedPhrases += 1
                firstUnsupportedPhrase = firstUnsupportedPhrase ?? phrase
                continue
            }
            acceptedPhrases.append(phrase)
            tokenizedPhrases.append(tokens)
            variativeRepresentations.append(representation)
            acceptedWeights.append(phraseWeights?[phraseIndex] ?? 1)
        }
        guard !acceptedPhrases.isEmpty else {
            throw PhraseBoostingError.untokenizablePhrase(firstUnsupportedPhrase ?? "")
        }

        let graph = try PhraseBoostingGraph(
            phrases: tokenizedPhrases.indices.map {
                PhraseGraphInput(
                    tokens: tokenizedPhrases[$0], variative: variativeRepresentations[$0],
                    weight: acceptedWeights[$0])
            },
            config: PhraseGraphConfiguration(
                contextScore: config.contextScore, depthScaling: config.depthScaling,
                unknownScore: config.unknownScore,
                variativeScoringTemperature: config.variativeScoringTemperature,
                penalizeSubsplits: config.penalizeSubsplits))
        self.phrases = acceptedPhrases
        self.tokenizedPhrases = tokenizedPhrases
        self.config = config
        self.skippedPhraseCount = skippedPhrases
        self.blankID = blankID
        self.graph = graph
        self.maximumRootTransitionScore = graph.maximumRootTransitionScore
        self.formattingPhraseIndices = Set(
            acceptedPhrases.indices.filter {
                config.caseInsensitive && acceptedPhrases[$0].lowercased() != acceptedPhrases[$0]
            })
        self.phraseEndBoundaryTokens = Set(
            vocabulary.compactMap { token, piece in
                if piece.hasPrefix(ASRConstants.sentencePieceWordBoundary) { return token }
                guard let scalar = piece.unicodeScalars.first else { return token }
                return CharacterSet.alphanumerics.contains(scalar) ? nil : token
            })
    }

    /// Re-select a non-blank greedy token after shallow fusion with the phrase graph.
    ///
    /// NeMo first decides blank versus non-blank from the unmodified acoustic model. If the
    /// result is non-blank, fusion is allowed to choose another non-blank vocabulary token.
    /// The caller enforces that blank-category decision before invoking this method.
    func select(baseToken: Int, acousticScores: [Float], state: Int) -> Selection {
        guard baseToken >= 0, baseToken < blankID, baseToken < acousticScores.count else {
            return Selection(token: baseToken, nextState: state, replacementProbability: nil)
        }

        let baseTransition = transition(from: state, token: baseToken)
        guard config.alpha > 0 else {
            return Selection(
                token: baseToken,
                nextState: baseTransition.nextState,
                replacementProbability: nil
            )
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
        let replacementProbability =
            bestToken == baseToken
            ? nil
            : tokenProbability(for: bestToken, acousticScores: acousticScores)
        return Selection(
            token: bestToken,
            nextState: bestTransition.nextState,
            replacementProbability: replacementProbability
        )
    }

    /// RNNTJoint may normalize token and duration logits together, while the
    /// optimized JointDecision model reports a token-only softmax probability.
    /// Renormalizing through the blank token keeps replacement confidences on
    /// the same scale as the primary decoder without rerunning either model.
    private func tokenProbability(for token: Int, acousticScores: [Float]) -> Float? {
        let tokenCount = min(blankID + 1, acousticScores.count)
        guard token >= 0, token < tokenCount else { return nil }

        var maximum = -Float.infinity
        for index in 0..<tokenCount {
            maximum = max(maximum, acousticScores[index])
        }
        guard maximum.isFinite else { return nil }

        var denominator = 0.0
        for index in 0..<tokenCount {
            denominator += Foundation.exp(Double(acousticScores[index] - maximum))
        }
        guard denominator.isFinite, denominator > 0 else { return nil }

        return Float(
            Foundation.exp(Double(acousticScores[token] - maximum)) / denominator
        )
    }

    func matchingPhrases(in tokens: [Int]) -> [String] {
        var matched = Set<Int>()
        var state = rootState
        for (index, token) in tokens.enumerated() {
            state = transition(from: state, token: token).nextState
            guard isPhraseEndBoundary(at: index + 1, tokens: tokens) else { continue }
            for phraseIndex in graph.matchingPhraseIndices(at: state) {
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

        while index < tokens.count, let child = graph.child(from: state, token: tokens[index]) {
            state = child
            index += 1
            guard isPhraseEndBoundary(at: index, tokens: tokens) else { continue }
            for phraseIndex in graph.terminalPhraseIndices(at: state)
            where formattingPhraseIndices.contains(phraseIndex) {
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

    /// Internal so tests can compare weighted graphs without driving the decoder.
    func transition(from originalState: Int, token: Int) -> PhraseBoostingGraph.Transition {
        graph.transition(from: originalState, token: token)
    }

}

/// SentencePiece BPE encoding reconstructed from the ordered Parakeet vocabulary.
///
/// Parakeet v2's piece ID is its BPE merge rank (the bundled NeMo `.vocab` score is
/// `-(id - 1)`). This produces the same token IDs without bundling a second tokenizer asset.
private struct ParakeetSentencePieceBPETokenizer {
    private let tokenToID: [String: Int]
    private let variants: VariativeBPEVocabulary

    init(vocabulary: [Int: String], blankID: Int) {
        let usable = vocabulary.filter { $0.key >= 0 && $0.key < blankID && $0.value != "<unk>" }
        var byPiece: [String: Int] = [:]
        for (id, piece) in usable { byPiece[piece] = min(id, byPiece[piece] ?? id) }
        tokenToID = byPiece
        variants = VariativeBPEVocabulary(vocabulary: usable, segmentation: .graphemes)
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

    func variativeRepresentation(for text: String) -> VariativeBPERepresentation? {
        guard let ids = encode(text) else { return nil }
        return variants.representation(for: ids)
    }
}
