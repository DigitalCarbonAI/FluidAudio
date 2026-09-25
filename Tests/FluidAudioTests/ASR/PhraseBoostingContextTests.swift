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

    func testCaseInsensitivePhraseUsesLowercaseTokensWhenOriginalCasingIsUnavailable() throws {
        var accentedVocabulary = vocabulary
        accentedVocabulary[896] = "ó"
        let context = try PhraseBoostingContext(
            phrases: ["Óla"],
            vocabulary: accentedVocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig()
        )

        XCTAssertEqual(context.phrases, ["Óla"])
        XCTAssertEqual(context.tokenizedPhrases, [[819, 896, 829, 823]])
        XCTAssertEqual(context.skippedPhraseCount, 0)
        XCTAssertEqual(context.matchingPhrases(in: context.tokenizedPhrases[0]), ["Óla"])
    }

    func testVariativeBPENewEndpointUsesSharedSourcePotential() throws {
        // This compact graph forces a merged-token alternative to reuse a higher-scored
        // source state. NeMo repairs the following newly created endpoint from that actual
        // source potential rather than the phrase-local accumulated score.
        let sharedStateVocabulary: [Int: String] = [
            0: "<unk>",
            1: "ba",
            2: "▁b",
            3: "ab",
            4: "▁",
            5: "a",
            6: "b",
            7: "c",
        ]
        let context = try PhraseBoostingContext(
            phrases: ["cabb", "baaa", "bbb"],
            vocabulary: sharedStateVocabulary,
            blankID: 16,
            config: PhraseBoostingConfig(variativeScoringTemperature: 0)
        )

        var prefixScores = Array(repeating: Float(-100), count: 17)
        prefixScores[2] = 0
        let prefix = context.select(
            baseToken: 2,
            acousticScores: prefixScores,
            state: context.rootState
        )
        XCTAssertEqual(prefix.token, 2)

        var continuationScores = Array(repeating: Float(-100), count: 17)
        continuationScores[5] = 0
        continuationScores[6] = -0.1
        let continuation = context.select(
            baseToken: 5,
            acousticScores: continuationScores,
            state: prefix.nextState
        )

        XCTAssertEqual(continuation.token, 6)
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

    func testUnsupportedPhraseCanBeSkippedWithoutDiscardingSupportedPhrases() throws {
        let context = try PhraseBoostingContext(
            phrases: ["Nguyễn", "codex"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(),
            skipUnsupportedPhrases: true
        )

        XCTAssertEqual(context.phrases, ["codex"])
        XCTAssertEqual(context.skippedPhraseCount, 1)
        XCTAssertEqual(context.matchingPhrases(in: context.tokenizedPhrases[0]), ["codex"])
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
            phraseBoostingFailureReason: .predictionFailed
        )

        let copied = failed.withRescoring(text: "Codex", detected: nil, applied: nil)

        XCTAssertEqual(copied.phraseBoostingFailed, true)
        XCTAssertEqual(copied.phraseBoostingFailureReason, .predictionFailed)
    }

    func testOnlyJointFailuresRequestOptionalResourceRepair() {
        XCTAssertFalse(PhraseBoostingFailureReason.workspaceUnavailable.requiresResourceRepair)
        XCTAssertTrue(PhraseBoostingFailureReason.jointUnavailable.requiresResourceRepair)
        XCTAssertTrue(PhraseBoostingFailureReason.predictionFailed.requiresResourceRepair)
    }

    // MARK: Phrase weights

    func testPhraseWeightsRejectMismatchedCountAndInvalidValues() {
        for weights: [Float] in [[1], [1, 1, 1], [1, 0], [1, -0.5], [1, .nan], [1, .infinity]] {
            XCTAssertThrowsError(
                try PhraseBoostingContext(
                    phrases: ["codex", "claude code"],
                    vocabulary: vocabulary,
                    blankID: 1_024,
                    config: PhraseBoostingConfig(),
                    phraseWeights: weights
                )
            ) { error in
                XCTAssertEqual(error as? PhraseBoostingError, .invalidConfiguration)
            }
        }
    }

    func testPhraseWeightsStayAlignedWhenUnsupportedPhrasesAreSkipped() throws {
        let skipped = try PhraseBoostingContext(
            phrases: ["Nguyễn", "codex", "claude code"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(),
            phraseWeights: [1, 0.5, 1],
            skipUnsupportedPhrases: true
        )
        let direct = try PhraseBoostingContext(
            phrases: ["codex", "claude code"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(),
            phraseWeights: [0.5, 1]
        )
        let misaligned = try PhraseBoostingContext(
            phrases: ["codex", "claude code"],
            vocabulary: vocabulary,
            blankID: 1_024,
            config: PhraseBoostingConfig(),
            phraseWeights: [1, 0.5]
        )

        XCTAssertEqual(skipped.phrases, direct.phrases)
        var differsFromMisaligned = false
        var state = skipped.rootState
        for token in skipped.tokenizedPhrases[0] {
            let actual = skipped.transition(from: state, token: token)
            let expected = direct.transition(from: state, token: token)
            XCTAssertEqual(actual.score, expected.score)
            XCTAssertEqual(actual.nextState, expected.nextState)
            if actual.score != misaligned.transition(from: state, token: token).score {
                differsFromMisaligned = true
            }
            state = actual.nextState
        }
        XCTAssertTrue(differsFromMisaligned)
    }

    func testUnitPhraseWeightsMatchOmittedWeights() throws {
        let phrases = ["codex", "claude code"]
        let omitted = try PhraseBoostingContext(
            phrases: phrases, vocabulary: vocabulary, blankID: 1_024, config: PhraseBoostingConfig())
        let unit = try PhraseBoostingContext(
            phrases: phrases, vocabulary: vocabulary, blankID: 1_024, config: PhraseBoostingConfig(),
            phraseWeights: [1, 1])
        for state in 0..<64 {
            for token in [-1, 7, 15, 16, 115, 287, 328, 471, 819, 820, 823, 850, 853] {
                let left = unit.transition(from: state, token: token)
                let right = omitted.transition(from: state, token: token)
                XCTAssertEqual(left.score.bitPattern, right.score.bitPattern)
                XCTAssertEqual(left.nextState, right.nextState)
            }
        }
    }

    /// A weight of r under fusion strength a must choose exactly like weight 1 under r * a,
    /// so a screen scale is a per-phrase alpha and nothing else.
    func testPhraseWeightActsLikeAScaledAlphaForThatPhrase() throws {
        func context(weight: Float, alpha: Float) throws -> PhraseBoostingContext {
            try PhraseBoostingContext(
                phrases: ["codex"],
                vocabulary: vocabulary,
                blankID: 1_024,
                config: PhraseBoostingConfig(alpha: alpha),
                phraseWeights: [weight]
            )
        }
        let weighted = try context(weight: 0.5, alpha: 4)
        let scaledAlpha = try context(weight: 1, alpha: 2)
        let unweighted = try context(weight: 1, alpha: 4)

        var weightMattered = false
        for deficit: Float in [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12] {
            var weightedState = weighted.rootState
            var scaledState = scaledAlpha.rootState
            for expected in weighted.tokenizedPhrases[0] {
                var scores = Array(repeating: Float(-100), count: 1_030)
                scores[7] = 0
                scores[expected] = -deficit
                let left = weighted.select(baseToken: 7, acousticScores: scores, state: weightedState)
                let right = scaledAlpha.select(baseToken: 7, acousticScores: scores, state: scaledState)
                XCTAssertEqual(left.token, right.token, "deficit \(deficit)")
                XCTAssertEqual(left.nextState, right.nextState)
                let full = unweighted.select(baseToken: 7, acousticScores: scores, state: weightedState)
                if full.token != left.token { weightMattered = true }
                weightedState = left.nextState
                scaledState = right.nextState
            }
        }
        XCTAssertTrue(weightMattered)
    }
}
