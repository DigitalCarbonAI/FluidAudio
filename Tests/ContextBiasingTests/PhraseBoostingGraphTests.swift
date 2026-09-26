import XCTest
@testable import ContextBiasing

final class PhraseBoostingGraphTests: XCTestCase {
    func testCandidateRowsEqualFullVocabularyTraversalAcrossBackoffAndCompletedSuffixes() throws {
        let graph = try PhraseBoostingGraph(
            phrases: [
                .init(tokens: [1, 2, 3]), .init(tokens: [2, 3]), .init(tokens: [1, 4]),
            ], config: .init(unknownScore: 0.2))
        var state = graph.rootState
        for token in [1, 2, 3, 7, 1, 2, 4, 1, 4, 2, 3] {
            let candidates = Set(graph.candidateTokens(from: state))
            let fallback = graph.transition(from: state, token: -1)
            for candidate in 0..<10 where !candidates.contains(candidate) {
                let actual = graph.transition(from: state, token: candidate)
                XCTAssertEqual(actual.score, fallback.score, accuracy: 0.000001)
                XCTAssertEqual(actual.nextState, fallback.nextState)
            }
            state = graph.transition(from: state, token: token).nextState
        }
        XCTAssertEqual(Set(graph.matchingPhraseIndices(at: state)), [1])
    }

    func testRejectsMalformedRepresentationsAndNonfiniteConfiguration() throws {
        for configuration in [PhraseGraphConfiguration(contextScore: .infinity), .init(unknownScore: -1)] {
            XCTAssertThrowsError(try PhraseBoostingGraph(phrases: [.init(tokens: [1])], config: configuration))
        }
        for representation in [
            VariativeBPERepresentation(canonicalLengths: [2], tokenGroups: [[.init(token: 1, length: 1)]]),
            .init(canonicalLengths: [1], tokenGroups: [[.init(token: 1, length: 2)]]),
            .init(canonicalLengths: [1], tokenGroups: [[]]),
            .init(canonicalLengths: [Int.max, 1], tokenGroups: []),
        ] {
            XCTAssertThrowsError(try PhraseBoostingGraph(phrases: [.init(tokens: [1], variative: representation)]))
        }
        XCTAssertThrowsError(try PhraseBoostingGraph(phrases: [.init(tokens: [])]))
        XCTAssertThrowsError(try PhraseBoostingGraph(phrases: []))
    }

    func testPreparationCancellation() {
        XCTAssertThrowsError(try PhraseBoostingGraph(phrases: [.init(tokens: [1])], shouldCancel: { true })) {
            XCTAssertTrue($0 is CancellationError)
        }
    }

    func testScalarSegmentationPreservesCombiningCharactersWithoutChangingLegacyMode() throws {
        let vocabulary = [0: "a", 1: "\u{301}", 2: "a\u{301}", 3: "A", 4: "A\u{301}"]
        let scalar = VariativeBPEVocabulary(vocabulary: vocabulary)
        let legacy = VariativeBPEVocabulary(vocabulary: vocabulary, segmentation: .graphemes)
        XCTAssertEqual(try XCTUnwrap(scalar.representation(for: [2])).canonicalLengths, [2])
        XCTAssertEqual(try XCTUnwrap(legacy.representation(for: [2])).canonicalLengths, [1])
    }

    // MARK: Phrase weights

    private let tokenRange = -1..<12

    private func variative(weight: Float = 1) -> PhraseGraphInput {
        // Canonical "ab" + "c", with a merged "ab" token (7) and an uppercase "a" (6).
        .init(
            tokens: [7, 8],
            variative: .init(
                canonicalLengths: [2, 1],
                tokenGroups: [
                    [.init(token: 5, length: 1), .init(token: 6, length: 1)],
                    [.init(token: 9, length: 1), .init(token: 7, length: 2)],
                    [.init(token: 8, length: 1)],
                ]),
            weight: weight)
    }

    private func assertEveryTransition(
        _ lhs: PhraseBoostingGraph, _ rhs: PhraseBoostingGraph,
        file: StaticString = #filePath, line: UInt = #line,
        _ scoreRelation: (Float, Float) -> Bool
    ) {
        XCTAssertEqual(lhs.stateCount, rhs.stateCount, file: file, line: line)
        for state in 0..<lhs.stateCount {
            for token in tokenRange {
                let left = lhs.transition(from: state, token: token)
                let right = rhs.transition(from: state, token: token)
                XCTAssertEqual(left.nextState, right.nextState, file: file, line: line)
                XCTAssertTrue(
                    scoreRelation(left.score, right.score),
                    "state \(state) token \(token): \(left.score) vs \(right.score)",
                    file: file, line: line)
            }
        }
    }

    func testHalfWeightHalvesEveryGreedyAndVariativeTransitionExactly() throws {
        let greedy: [[Int]] = [[1, 2, 3], [2, 3], [1, 4]]
        let full = try PhraseBoostingGraph(phrases: greedy.map { .init(tokens: $0) })
        let half = try PhraseBoostingGraph(phrases: greedy.map { .init(tokens: $0, weight: 0.5) })
        // Scaling by a power of two is exact in binary floating point, so forward rewards and
        // backoff refunds must halve bit for bit.
        assertEveryTransition(half, full) { $0 == $1 * 0.5 }

        let fullVariative = try PhraseBoostingGraph(phrases: [variative()])
        let halfVariative = try PhraseBoostingGraph(phrases: [variative(weight: 0.5)])
        assertEveryTransition(halfVariative, fullVariative) { $0 == $1 * 0.5 }
    }

    func testUnitWeightBuildsTheSameGraphAsOmittedWeight() throws {
        let omitted = try PhraseBoostingGraph(phrases: [
            .init(tokens: [1, 2, 3]), .init(tokens: [2, 3]), .init(tokens: [1, 4]),
            .init(tokens: [7, 8], variative: variative().variative),
        ])
        let unit = try PhraseBoostingGraph(phrases: [
            .init(tokens: [1, 2, 3], weight: 1), .init(tokens: [2, 3], weight: 1),
            .init(tokens: [1, 4], weight: 1), variative(weight: 1),
        ])
        assertEveryTransition(unit, omitted) { $0.bitPattern == $1.bitPattern }
        XCTAssertEqual(unit.maximumRootTransitionScore, omitted.maximumRootTransitionScore)
        for state in 0..<unit.stateCount {
            XCTAssertEqual(unit.isTerminal(state), omitted.isTerminal(state))
            XCTAssertEqual(unit.terminalPhraseIndices(at: state), omitted.terminalPhraseIndices(at: state))
            XCTAssertEqual(unit.matchingPhraseIndices(at: state), omitted.matchingPhraseIndices(at: state))
        }
    }

    func testMixedWeightSharedPrefixIsOrderIndependentAndTakesTheLargerReward() throws {
        let strong = PhraseGraphInput(tokens: [1, 2, 3], weight: 1)
        let weak = PhraseGraphInput(tokens: [1, 4, 5], weight: 0.5)
        let strongFirst = try PhraseBoostingGraph(phrases: [strong, weak])
        let weakFirst = try PhraseBoostingGraph(phrases: [weak, strong])
        assertEveryTransition(strongFirst, weakFirst) { $0.bitPattern == $1.bitPattern }

        // The shared first token earns the strong phrase's reward in both orders.
        let unweighted = try PhraseBoostingGraph(phrases: [.init(tokens: [1, 2, 3])])
        XCTAssertEqual(
            weakFirst.transition(from: weakFirst.rootState, token: 1).score,
            unweighted.transition(from: unweighted.rootState, token: 1).score)

        // Phrase indices still refer to the caller's order.
        var state = weakFirst.rootState
        for token in [1, 4, 5] { state = weakFirst.transition(from: state, token: token).nextState }
        XCTAssertEqual(weakFirst.terminalPhraseIndices(at: state), [0])
    }

    func testIncompleteWeightedPrefixRefundsItsWholeRewardOnBackoff() throws {
        for phrases in [
            [PhraseGraphInput(tokens: [1, 2, 3], weight: 1), .init(tokens: [1, 4, 5], weight: 0.5)],
            [.init(tokens: [1, 4, 5], weight: 0.5), .init(tokens: [1, 2, 3], weight: 1)],
            [variative(weight: 0.5), .init(tokens: [5, 6, 1], weight: 1)],
        ] {
            let graph = try PhraseBoostingGraph(phrases: phrases)
            for prefix in [[1], [1, 4], [1, 2], [5], [5, 9], [7]] {
                var state = graph.rootState
                var total: Float = 0
                for token in prefix + [11] {
                    let transition = graph.transition(from: state, token: token)
                    total += transition.score
                    state = transition.nextState
                }
                XCTAssertEqual(state, graph.rootState)
                XCTAssertEqual(total, 0, accuracy: 0.000_01, "prefix \(prefix)")
            }
        }
    }

    func testRejectsNonpositiveAndNonfiniteWeights() {
        for weight: Float in [0, -1, .nan, .infinity, -.infinity] {
            XCTAssertThrowsError(
                try PhraseBoostingGraph(phrases: [.init(tokens: [1]), .init(tokens: [2], weight: weight)])
            ) { XCTAssertEqual($0 as? PhraseGraphError, .invalidConfiguration) }
        }
    }
}
