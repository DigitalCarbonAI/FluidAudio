import XCTest

@testable import FluidAudio

final class PhraseBoostingContextTests: XCTestCase {
    private let vocabulary: [Int: String] = [
        0: "<unk>",
        15: "▁C",
        16: "▁c",
        115: "od",
        287: "ud",
        328: "▁cl",
        471: "co",
        819: "▁",
        820: "e",
        822: "o",
        823: "a",
        829: "l",
        830: "d",
        831: "u",
        832: "c",
        850: "x",
        853: "C",
        854: "O",
        855: "D",
        856: "E",
        857: "X",
    ]

    func testParakeetTokenizerMatchesNeMoSentencePieceFixture() throws {
        let context = try PhraseBoostingContext(
            phrases: ["claude code", "codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )

        XCTAssertEqual(context.tokenizedPhrases[0], [328, 823, 287, 820, 16, 115, 820])
        XCTAssertEqual(context.tokenizedPhrases[1], [16, 115, 820, 850])
    }

    func testTurboBiasPreservesBlankCategoryAtCallBoundary() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 4)
        )

        let selection = context.select(
            baseToken: 1_024,
            acousticScores: Array(repeating: -100, count: 1_030),
            state: context.rootState
        )

        XCTAssertEqual(selection.token, 1_024)
        XCTAssertEqual(selection.nextState, context.rootState)
    }

    func testZeroAlphaIsByteStableEvenWhenFullJointPrefersAnotherToken() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 0)
        )
        var scores = Array(repeating: Float(-100), count: 1_030)
        scores[7] = -20
        scores[16] = 0

        let selection = context.select(baseToken: 7, acousticScores: scores, state: context.rootState)

        XCTAssertEqual(selection.token, 7)
    }

    func testSeparateJointCannotReplaceBaseWithEquallyUnboostedToken() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 2)
        )
        var scores = Array(repeating: Float(-100), count: 1_030)
        scores[7] = -20
        scores[8] = 0

        let selection = context.select(baseToken: 7, acousticScores: scores, state: context.rootState)

        XCTAssertEqual(selection.token, 7)
    }

    func testEqualRootRewardsAdvanceStateWithoutFullVocabularyScores() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 1, unknownScore: 1)
        )

        let unrelated = try XCTUnwrap(
            context.selectionWithoutAcousticScores(baseToken: 7, state: context.rootState))
        XCTAssertEqual(unrelated.token, 7)
        XCTAssertEqual(unrelated.nextState, context.rootState)

        let prefix = try XCTUnwrap(
            context.selectionWithoutAcousticScores(baseToken: 16, state: context.rootState))
        XCTAssertEqual(prefix.token, 16)
        XCTAssertNotEqual(prefix.nextState, context.rootState)
        XCTAssertNil(
            context.selectionWithoutAcousticScores(baseToken: 115, state: prefix.nextState))
        XCTAssertNil(
            context.selectionWithoutAcousticScores(baseToken: 819, state: context.rootState))
    }

    func testZeroAlphaNeverNeedsFullVocabularyScores() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 0)
        )
        let prefix = try XCTUnwrap(
            context.selectionWithoutAcousticScores(baseToken: 16, state: context.rootState))

        XCTAssertNotNil(
            context.selectionWithoutAcousticScores(baseToken: 115, state: prefix.nextState))
    }

    func testHighConfidenceRootTokenCannotBeOvertakenByMaximumBoost() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 1)
        )

        XCTAssertNotNil(
            context.selectionWithoutAcousticScores(
                baseToken: 7,
                baseProbability: 0.8,
                state: context.rootState
            )
        )
        XCTAssertNil(
            context.selectionWithoutAcousticScores(
                baseToken: 7,
                baseProbability: 0.7,
                state: context.rootState
            )
        )
    }

    func testDefaultUnknownScoreCannotReplaceEarlyVariativePrefixWithUnrelatedToken() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )
        var scores = Array(repeating: Float(-100), count: 1_030)
        scores[819] = -20
        scores[7] = 0

        let selection = context.select(
            baseToken: 819,
            acousticScores: scores,
            state: context.rootState
        )

        XCTAssertEqual(selection.token, 819)
    }

    func testTurboBiasCanPromoteAndCompleteAMultiTokenPhrase() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 1)
        )
        var state = context.rootState

        for expected in context.tokenizedPhrases[0] {
            var scores = Array(repeating: Float(-100), count: 1_030)
            scores[7] = 0
            scores[expected] = -0.5
            let selection = context.select(baseToken: 7, acousticScores: scores, state: state)
            XCTAssertEqual(selection.token, expected)
            state = selection.nextState
        }

        XCTAssertEqual(
            context.matchingPhrases(in: context.tokenizedPhrases[0]),
            ["codex"]
        )
    }

    func testReplacementReportsTokenOnlyAcousticProbability() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 1)
        )
        let replacementToken = try XCTUnwrap(context.tokenizedPhrases[0].first)
        var scores = Array(repeating: Float(-100), count: 1_030)
        scores[7] = 0
        scores[replacementToken] = -0.5
        scores[1_024] = -1

        let replacement = context.select(
            baseToken: 7,
            acousticScores: scores,
            state: context.rootState
        )
        let expected = Float(
            Foundation.exp(-0.5)
                / (Foundation.exp(0) + Foundation.exp(-0.5) + Foundation.exp(-1))
        )

        XCTAssertEqual(replacement.token, replacementToken)
        XCTAssertEqual(try XCTUnwrap(replacement.replacementProbability), expected, accuracy: 1e-6)

        scores[replacementToken] = -10
        let unchanged = context.select(
            baseToken: 7,
            acousticScores: scores,
            state: context.rootState
        )
        XCTAssertEqual(unchanged.token, 7)
        XCTAssertNil(unchanged.replacementProbability)
    }

    func testLowercasePhraseAlsoMatchesTitleCaseTokensForParakeetV2() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )

        XCTAssertEqual(context.matchingPhrases(in: [15, 115, 820, 850]), ["codex"])
        XCTAssertEqual(
            context.formattedText(in: [15, 115, 820, 850], vocabulary: vocabulary),
            "Codex"
        )
    }

    func testVariativeBPEMatchesAllCapsAndCharacterSplit() throws {
        let context = try PhraseBoostingContext(
            phrases: ["Codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )
        let allCapsCharacterTokens = [819, 853, 854, 855, 856, 857]
        let lowercaseCharacterTokens = [819, 832, 822, 830, 820, 850]

        XCTAssertEqual(context.matchingPhrases(in: allCapsCharacterTokens), ["Codex"])
        XCTAssertEqual(context.matchingPhrases(in: lowercaseCharacterTokens), ["Codex"])
        XCTAssertEqual(
            context.formattedText(in: allCapsCharacterTokens, vocabulary: vocabulary),
            "Codex"
        )
    }

    func testAlternativeLowercasePathRestoresDeliberateDictionaryCasing() throws {
        let context = try PhraseBoostingContext(
            phrases: ["Codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )

        XCTAssertEqual(
            context.formattedText(in: [16, 115, 820, 850], vocabulary: vocabulary),
            "Codex"
        )

        let disabled = try PhraseBoostingContext(
            phrases: ["Codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 0)
        )
        XCTAssertEqual(
            disabled.formattedText(in: [16, 115, 820, 850], vocabulary: vocabulary),
            "codex"
        )
    }

    func testPhrasePrefixDoesNotReformatOrCountAsAWholeWordMatch() throws {
        let context = try PhraseBoostingContext(
            phrases: ["Cod"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )
        let codex = [16, 115, 820, 850]

        XCTAssertEqual(context.formattedText(in: codex, vocabulary: vocabulary), "codex")
        XCTAssertEqual(context.matchingPhrases(in: codex), [])
    }

    func testAutomatonReportsBothPhraseAndCompletedSuffix() throws {
        let context = try PhraseBoostingContext(
            phrases: ["claude code", "code"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )

        XCTAssertEqual(
            context.matchingPhrases(in: context.tokenizedPhrases[0]),
            ["claude code", "code"]
        )
    }

    func testFormattingAutomatonPrefersLongestStyledPhrase() throws {
        let context = try PhraseBoostingContext(
            phrases: ["Claude Code", "Code"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )
        let lowercaseTokens = try PhraseBoostingContext(
            phrases: ["claude code"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        ).tokenizedPhrases[0]

        XCTAssertEqual(
            context.formattedText(in: lowercaseTokens, vocabulary: vocabulary),
            "Claude Code"
        )
    }

    func testIncompletePhraseRewardIsRemovedOnBackoff() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 1)
        )
        var firstScores = Array(repeating: Float(-100), count: 1_030)
        firstScores[7] = 0
        firstScores[16] = -0.5
        let prefix = context.select(
            baseToken: 7,
            acousticScores: firstScores,
            state: context.rootState
        )
        XCTAssertEqual(prefix.token, 16)

        var mismatchScores = Array(repeating: Float(-100), count: 1_030)
        mismatchScores[7] = 0
        mismatchScores[115] = -10
        let mismatch = context.select(
            baseToken: 7,
            acousticScores: mismatchScores,
            state: prefix.nextState
        )

        XCTAssertEqual(mismatch.token, 7)
        XCTAssertEqual(mismatch.nextState, context.rootState)
    }

    func testCompletedPhraseRewardSurvivesBackoff() throws {
        let context = try PhraseBoostingContext(
            phrases: ["codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(alpha: 1)
        )
        var state = context.rootState
        for token in context.tokenizedPhrases[0] {
            var scores = Array(repeating: Float(-100), count: 1_030)
            scores[token] = 0
            let selection = context.select(baseToken: token, acousticScores: scores, state: state)
            state = selection.nextState
        }

        var scores = Array(repeating: Float(-100), count: 1_030)
        scores[7] = 0
        let next = context.select(baseToken: 7, acousticScores: scores, state: state)

        XCTAssertEqual(next.token, 7)
        XCTAssertEqual(next.nextState, context.rootState)
    }

    func testUnknownCharactersFailPreparationInsteadOfBoostingUnknownToken() {
        XCTAssertThrowsError(
            try PhraseBoostingContext(
                phrases: ["Nguyễn"],
                vocabulary: vocabulary,
                blankID: 1_024,
                config: PhraseBoostingConfig()
            )
        ) { error in
            XCTAssertEqual(error as? PhraseBoostingError, .untokenizablePhrase("Nguyễn"))
        }
    }

    func testRejectsNegativeUnknownScore() {
        XCTAssertThrowsError(
            try PhraseBoostingContext(
                phrases: ["codex"],
                vocabulary: vocabulary,
                blankID: 1_024,
                config: PhraseBoostingConfig(unknownScore: -1)
            )
        ) { error in
            XCTAssertEqual(error as? PhraseBoostingError, .invalidConfiguration)
        }
    }

    func testPhraseFailureMetadataSurvivesRescoringCopy() {
        let failed = ASRResult(
            text: "codex",
            confidence: 1,
            duration: 1,
            processingTime: 0.1,
            phraseBoostingFailed: true
        )

        let copied = failed.withRescoring(text: "Codex", detected: nil, applied: nil)

        XCTAssertEqual(copied.phraseBoostingFailed, true)
    }
}
