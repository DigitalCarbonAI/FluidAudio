import Foundation
import XCTest

@testable import FluidAudio

final class PhraseBoostingIntegrationTests: XCTestCase {
    func testRealParakeetV2PhraseBoostingWhenExplicitlyEnabled() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["FLUID_AUDIO_TURBOBIAS_MODEL_DIR"],
            let audioPath = environment["FLUID_AUDIO_TURBOBIAS_AUDIO"],
            let phrasesValue = environment["FLUID_AUDIO_TURBOBIAS_PHRASES"]
        else {
            throw XCTSkip(
                "Set FLUID_AUDIO_TURBOBIAS_MODEL_DIR, FLUID_AUDIO_TURBOBIAS_AUDIO, and "
                    + "FLUID_AUDIO_TURBOBIAS_PHRASES to run the real-model check.")
        }

        let phrases = phrasesValue.split(separator: "|").map(String.init)
        let alpha = environment["FLUID_AUDIO_TURBOBIAS_ALPHA"].flatMap(Float.init) ?? 1
        let unknownScore = environment["FLUID_AUDIO_TURBOBIAS_UNKNOWN_SCORE"].flatMap(Float.init) ?? 0
        let models = try AsrModels.loadDirect(
            from: URL(fileURLWithPath: modelPath),
            version: .v2)
        let manager = AsrManager(
            config: ASRConfig(
                tdtConfig: TdtConfig(blankId: AsrModelVersion.v2.blankId),
                encoderHiddenSize: AsrModelVersion.v2.encoderHiddenSize),
            models: models)
        let phraseJointPath = environment["FLUID_AUDIO_TURBOBIAS_JOINT_DIR"] ?? modelPath
        try await manager.loadPhraseBoostingJoint(
            from: URL(fileURLWithPath: phraseJointPath))
        let supportsPhraseBoosting = await manager.supportsPhraseBoosting
        XCTAssertTrue(supportsPhraseBoosting)

        let audioURL = URL(fileURLWithPath: audioPath)
        let baseline = try await transcribe(manager: manager, audioURL: audioURL)
        let zeroContext = try await manager.makePhraseBoostingContext(
            phrases: phrases,
            config: PhraseBoostingConfig(alpha: 0))
        let zero = try await transcribe(
            manager: manager, audioURL: audioURL, phraseBoosting: zeroContext)
        XCTAssertEqual(zero.result.text, baseline.result.text)
        XCTAssertEqual(zero.result.tokenTimings?.map(\.tokenId), baseline.result.tokenTimings?.map(\.tokenId))

        let context = try await manager.makePhraseBoostingContext(
            phrases: phrases,
            config: PhraseBoostingConfig(alpha: alpha, unknownScore: unknownScore))
        let boosted = try await transcribe(
            manager: manager, audioURL: audioURL, phraseBoosting: context)

        print("TURBOBIAS_REAL_AUDIO_BEGIN")
        print("baseline_seconds=\(baseline.wallDuration) text=\(baseline.result.text)")
        print("alpha_zero_seconds=\(zero.wallDuration) text=\(zero.result.text)")
        print(
            "boosted_seconds=\(boosted.wallDuration) alpha=\(alpha) unknown_score=\(unknownScore) "
                + "text=\(boosted.result.text)")
        print("matched_terms=\(boosted.result.phraseBoostedTerms ?? [])")
        print("TURBOBIAS_REAL_AUDIO_END")

        if let expected = environment["FLUID_AUDIO_TURBOBIAS_EXPECTED"] {
            XCTAssertEqual(boosted.result.text, expected)
        }

        await manager.unloadPhraseBoostingJoint()
        let supportsAfterUnload = await manager.supportsPhraseBoosting
        XCTAssertFalse(supportsAfterUnload)
        try await manager.loadPhraseBoostingJoint(from: URL(fileURLWithPath: phraseJointPath))
        let supportsAfterReload = await manager.supportsPhraseBoosting
        XCTAssertTrue(supportsAfterReload)
    }

    private func transcribe(
        manager: AsrManager,
        audioURL: URL,
        phraseBoosting: PhraseBoostingContext? = nil
    ) async throws -> (result: ASRResult, wallDuration: TimeInterval) {
        let started = Date()
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let result = try await manager.transcribe(
            audioURL,
            decoderState: &state,
            phraseBoosting: phraseBoosting)
        return (result, Date().timeIntervalSince(started))
    }
}
