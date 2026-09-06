// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Graph portions Copyright 2023 Xiaomi Corp. (Wei Kang). Apache-2.0.
// Derived from NVIDIA NeMo Speech, Copyright NVIDIA Corporation (Apache-2.0).
// Original context graph: Copyright 2023 Xiaomi Corp. (Wei Kang), Apache-2.0.
import Foundation

/// Decoder-independent weighted phrase graph. Tokenizers and token selection live in adapters.
public struct PhraseBoostingGraph: Sendable {
    public struct Transition: Sendable {
        public let nextState: Int
        public let score: Float
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
    }

    private let nodes: [Node]
    public let config: PhraseGraphConfiguration
    public let maximumRootTransitionScore: Float
    public var rootState: Int { 0 }
    public var stateCount: Int { nodes.count }

    public init(
        phrases: [PhraseGraphInput], config: PhraseGraphConfiguration = .init(),
        shouldCancel: () -> Bool = { false }
    ) throws {
        try config.validate()
        guard !phrases.isEmpty else { throw PhraseGraphError.emptyPhrases }
        var buildingNodes = [Node()]
        for (index, phrase) in phrases.enumerated() {
            if shouldCancel() { throw CancellationError() }
            try phrase.validate()
            let terminal: Int
            if let representation = phrase.variative {
                terminal = Self.addVariativePhrase(representation, config: config, to: &buildingNodes)
            } else {
                terminal = Self.addGreedyPhrase(phrase.tokens, config: config, to: &buildingNodes)
            }
            buildingNodes[terminal].isEnd = true
            buildingNodes[terminal].phraseIndices.append(index)
        }
        for index in buildingNodes.indices {
            buildingNodes[index].outputPhraseIndices = buildingNodes[index].phraseIndices
        }
        Self.fillFailureLinks(in: &buildingNodes)
        guard buildingNodes.allSatisfy({ $0.nodeScore.isFinite }) else {
            throw PhraseGraphError.invalidConfiguration
        }
        self.nodes = buildingNodes
        self.config = config
        maximumRootTransitionScore = max(
            config.unknownScore,
            buildingNodes[0].children.values.map { buildingNodes[$0].nodeScore }.max() ?? 0)
    }

    public func child(from state: Int, token: Int) -> Int? {
        guard nodes.indices.contains(state) else { return nil }
        return nodes[state].children[token]
    }

    public func isTerminal(_ state: Int) -> Bool {
        nodes.indices.contains(state) && nodes[state].isEnd
    }

    public func terminalPhraseIndices(at state: Int) -> [Int] {
        nodes.indices.contains(state) ? nodes[state].phraseIndices : []
    }

    public func matchingPhraseIndices(at state: Int) -> [Int] {
        nodes.indices.contains(state) ? nodes[state].outputPhraseIndices : []
    }

    /// Only these tokens differ from the common unknown/backoff score for this state.
    /// Tensor adapters can build one score row without transferring acoustic logits to CPU.
    public func candidateTokens(from state: Int) -> [Int] {
        var current = nodes.indices.contains(state) ? state : rootState
        var tokens = Set<Int>()
        while true {
            tokens.formUnion(nodes[current].children.keys)
            if current == rootState { break }
            current = nodes[current].fail
        }
        return tokens.sorted()
    }
    private static func addGreedyPhrase(
        _ tokens: [Int],
        config: PhraseGraphConfiguration,
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
        _ representation: VariativeBPERepresentation,
        config: PhraseGraphConfiguration,
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

            if isPrimaryEndpoint[index] {
                let sourceIndex = statesByCanonicalPosition.count - primaryBackJumps[index]
                let primaryPotential =
                    nodes[statesByCanonicalPosition[sourceIndex]].nodeScore + primaryScores[index]
                nodes[nextState].nodeScore = max(nodes[nextState].nodeScore, primaryPotential)
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

    private static func score(depth: Int, config: PhraseGraphConfiguration) -> Float {
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

    public func transition(from originalState: Int, token: Int) -> Transition {
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
