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
}
