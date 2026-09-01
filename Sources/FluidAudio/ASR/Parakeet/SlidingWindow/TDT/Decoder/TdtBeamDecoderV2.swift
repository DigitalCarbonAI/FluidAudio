// The bounded TDT beam-search structure is derived from NVIDIA NeMo Speech's
// `BeamTDTInfer.default_beam_search`, Copyright NVIDIA Corporation, licensed
// under Apache License 2.0.
import CoreML
import Foundation

internal enum TdtPhraseBoostingBeamError: Error {
    case workspaceUnavailable
    case predictionFailed

    var failureReason: PhraseBoostingFailureReason {
        switch self {
        case .workspaceUnavailable: .workspaceUnavailable
        case .predictionFailed: .predictionFailed
        }
    }
}

internal enum TdtBeamSearchMath {
    static func logSoftmax(_ values: ArraySlice<Float>) -> [Float] {
        guard let maximum = values.max(), maximum.isFinite else {
            return Array(repeating: -.infinity, count: values.count)
        }
        var denominator = 0.0
        for value in values {
            denominator += Foundation.exp(Double(value - maximum))
        }
        guard denominator.isFinite, denominator > 0 else {
            return Array(repeating: -.infinity, count: values.count)
        }
        let logDenominator = Float(Foundation.log(denominator))
        return values.map { $0 - maximum - logDenominator }
    }

    static func topIndices(in values: [Float], limit: Int) -> [Int] {
        guard limit > 0 else { return [] }
        return values.indices.sorted {
            if values[$0] == values[$1] { return $0 < $1 }
            return values[$0] > values[$1]
        }.prefix(limit).map(\.self)
    }

    static func logAddExp(_ left: Float, _ right: Float) -> Float {
        let maximum = max(left, right)
        guard maximum.isFinite else { return maximum }
        return maximum
            + Float(
                Foundation.log(
                    Foundation.exp(Double(left - maximum))
                        + Foundation.exp(Double(right - maximum))
                )
            )
    }
}

/// A bounded Core ML port of NeMo's TDT beam search with TurboBias shallow fusion.
///
/// The full-vocabulary joint is evaluated for every active hypothesis. This is intentionally
/// separate from the optimized greedy decoder so callers without an explicit beam context keep
/// the exact existing scalar-joint path and allocation profile.
internal struct TdtBeamDecoderV2 {
    private struct HypothesisKey: Hashable {
        let history: [Int]
        let nextFrame: Int
        let phraseState: Int
    }

    private struct BeamHypothesis {
        var logScore: Float
        var history: [Int]
        var outputTokens: [Int]
        var timestamps: [Int]
        var durations: [Int]
        var confidences: [Float]
        var decoderState: TdtDecoderState
        var decoderProjection: MLMultiArray
        var phraseState: Int
        var nextFrame: Int
        var zeroDurationSymbols: Int

        var key: HypothesisKey {
            HypothesisKey(
                history: history,
                nextFrame: nextFrame,
                phraseState: phraseState
            )
        }
    }

    private struct TokenDurationExpansion {
        let token: PhraseBoostingContext.BeamTokenCandidate
        let durationIndex: Int
        let score: Float
    }

    private struct CachedDecoderStep {
        let state: TdtDecoderState
        let projection: MLMultiArray
    }

    private let config: ASRConfig
    private let modelInference = TdtModelInference()

    init(config: ASRConfig) {
        self.config = config
    }

    func decodeWithTimings(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        decoderModel: MLModel,
        phraseBoostingJointModel: MLModel,
        phraseBoostingContext: PhraseBoostingContext,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int,
        isLastChunk: Bool,
        globalFrameOffset: Int
    ) async throws -> TdtHypothesis {
        guard encoderSequenceLength > 1 else {
            return TdtHypothesis(decState: decoderState)
        }
        guard let beamWidth = phraseBoostingContext.config.decoder.beamWidth else {
            preconditionFailure("The beam decoder requires an explicit beam configuration.")
        }

        let encoderFrames = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: encoderSequenceLength,
            expectedHiddenSize: config.encoderHiddenSize
        )
        let initialFrame = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: decoderState.timeJump,
            contextFrameAdjustment: contextFrameAdjustment
        )
        let navigation = TdtFrameNavigation.initializeNavigationState(
            timeIndices: initialFrame,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames
        )
        guard navigation.activeMask else {
            return TdtHypothesis(decState: decoderState)
        }

        let workspace: PhraseBoostingJointWorkspace
        do {
            workspace = try PhraseBoostingJointWorkspace(
                encoderHiddenSize: config.encoderHiddenSize,
                decoderHiddenSize: ASRConstants.decoderHiddenSize,
                outputSize: config.tdtConfig.blankId + config.tdtConfig.durationBins.count + 1
            )
        } catch {
            throw TdtPhraseBoostingBeamError.workspaceUnavailable
        }

        let target = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let targetLength = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        targetLength[0] = 1

        let start: BeamHypothesis
        do {
            var startState = try TdtDecoderState(from: decoderState)
            if startState.lastToken == nil && startState.predictorOutput == nil {
                startState.hiddenState.resetData(to: 0)
                startState.cellState.resetData(to: 0)
            }

            let projection: MLMultiArray
            if let cached = startState.predictorOutput {
                projection = try modelInference.normalizeDecoderProjection(cached)
            } else {
                let primed = try modelInference.runDecoder(
                    token: startState.lastToken ?? config.tdtConfig.blankId,
                    state: startState,
                    model: decoderModel,
                    targetArray: target,
                    targetLengthArray: targetLength
                )
                startState = primed.newState
                guard let rawProjection = primed.output.featureValue(for: "decoder")?.multiArrayValue else {
                    throw ASRError.processingFailed("Beam decoder output is missing its projection")
                }
                projection = try modelInference.normalizeDecoderProjection(rawProjection)
            }
            startState.predictorOutput = projection
            start = BeamHypothesis(
                logScore: 0,
                history: [],
                outputTokens: [],
                timestamps: [],
                durations: [],
                confidences: [],
                decoderState: startState,
                decoderProjection: projection,
                phraseState: decoderState.phraseBoostingState ?? phraseBoostingContext.rootState,
                nextFrame: initialFrame,
                zeroDurationSymbols: 0
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as ASRError {
            throw error
        } catch {
            throw TdtPhraseBoostingBeamError.workspaceUnavailable
        }

        // The prediction network depends only on the emitted token history, not on the TDT
        // duration. The default beam commonly keeps several duration variants of the same
        // token path, so caching this result avoids repeating the large LSTM for each variant.
        var decoderCache: [[Int]: CachedDecoderStep] = [
            []: CachedDecoderStep(state: start.decoderState, projection: start.decoderProjection)
        ]
        var kept = [start]
        for frame in initialFrame..<navigation.effectiveSequenceLength {
            try Task.checkCancellation()
            var current = kept.filter { $0.nextFrame == frame }
            kept.removeAll { $0.nextFrame == frame }

            while !current.isEmpty {
                try Task.checkCancellation()
                current.sort(by: hypothesisRanksBefore)
                let hypothesis = current.removeFirst()

                let logits: [Float]
                do {
                    logits = try workspace.scores(
                        encoderFrames: encoderFrames,
                        timeIndex: frame,
                        preparedDecoderStep: hypothesis.decoderProjection,
                        model: phraseBoostingJointModel
                    )
                } catch {
                    throw TdtPhraseBoostingBeamError.predictionFailed
                }

                let tokenCount = config.tdtConfig.blankId + 1
                guard logits.count == tokenCount + config.tdtConfig.durationBins.count else {
                    throw TdtPhraseBoostingBeamError.predictionFailed
                }
                let tokenLogProbabilities = TdtBeamSearchMath.logSoftmax(logits[..<tokenCount])
                let durationLogProbabilities = TdtBeamSearchMath.logSoftmax(logits[tokenCount...])
                let tokenCandidates = phraseBoostingContext.beamTokenCandidates(
                    acousticLogProbabilities: tokenLogProbabilities,
                    state: hypothesis.phraseState,
                    limit: beamWidth
                )
                let durationIndices = TdtBeamSearchMath.topIndices(
                    in: durationLogProbabilities,
                    limit: min(beamWidth, durationLogProbabilities.count)
                )

                if hypothesis.history.count < config.tdtConfig.maxTokensPerChunk {
                    var expansions = [TokenDurationExpansion]()
                    expansions.reserveCapacity(tokenCandidates.count * durationIndices.count)
                    for token in tokenCandidates {
                        for durationIndex in durationIndices {
                            let duration = config.tdtConfig.durationBins[durationIndex]
                            guard
                                duration != 0
                                    || hypothesis.zeroDurationSymbols < config.tdtConfig.maxSymbolsPerStep
                            else { continue }
                            expansions.append(
                                TokenDurationExpansion(
                                    token: token,
                                    durationIndex: durationIndex,
                                    score: token.fusedLogProbability
                                        + durationLogProbabilities[durationIndex]
                                )
                            )
                        }
                    }
                    expansions.sort {
                        if $0.score == $1.score {
                            if $0.token.token == $1.token.token {
                                return $0.durationIndex < $1.durationIndex
                            }
                            return $0.token.token < $1.token.token
                        }
                        return $0.score > $1.score
                    }

                    for expansion in expansions.prefix(beamWidth) {
                        let duration = config.tdtConfig.durationBins[expansion.durationIndex]
                        let nextHistory = hypothesis.history + [expansion.token.token]
                        let decoderStep: CachedDecoderStep
                        if let cached = decoderCache[nextHistory] {
                            decoderStep = cached
                        } else {
                            var nextState = try TdtDecoderState(from: hypothesis.decoderState)
                            let decoderResult = try modelInference.runDecoder(
                                token: expansion.token.token,
                                state: nextState,
                                model: decoderModel,
                                targetArray: target,
                                targetLengthArray: targetLength
                            )
                            nextState = decoderResult.newState
                            guard
                                let rawProjection = decoderResult.output.featureValue(for: "decoder")?
                                    .multiArrayValue
                            else {
                                throw ASRError.processingFailed(
                                    "Beam decoder output is missing its projection"
                                )
                            }
                            let nextProjection = try modelInference.normalizeDecoderProjection(
                                rawProjection
                            )
                            nextState.predictorOutput = nextProjection
                            nextState.lastToken = expansion.token.token
                            decoderStep = CachedDecoderStep(
                                state: nextState,
                                projection: nextProjection
                            )
                            decoderCache[nextHistory] = decoderStep
                        }

                        let timestamp = frame + globalFrameOffset
                        var next = hypothesis
                        next.logScore += expansion.score
                        next.history = nextHistory
                        next.outputTokens.append(expansion.token.token)
                        next.timestamps.append(timestamp)
                        next.durations.append(duration)
                        next.confidences.append(
                            TdtDurationMapping.clampProbability(
                                Float(Foundation.exp(Double(expansion.token.acousticLogProbability)))
                            )
                        )
                        next.decoderState = decoderStep.state
                        next.decoderProjection = decoderStep.projection
                        next.phraseState = expansion.token.nextState
                        next.nextFrame = frame + duration
                        next.zeroDurationSymbols =
                            duration == 0
                            ? hypothesis.zeroDurationSymbols + 1 : 0
                        if duration == 0 {
                            current.append(next)
                        } else {
                            kept.append(next)
                        }
                    }
                }

                let blankLogProbability = tokenLogProbabilities[config.tdtConfig.blankId]
                for durationIndex in durationIndices {
                    let duration = config.tdtConfig.durationBins[durationIndex]
                    guard duration > 0 else { continue }
                    var blank = hypothesis
                    blank.logScore +=
                        blankLogProbability
                        + durationLogProbabilities[durationIndex]
                    blank.nextFrame = frame + duration
                    blank.zeroDurationSymbols = 0
                    kept.append(blank)
                }

                // NeMo's sequential reference leaves this zero-duration frontier
                // unbounded until future hypotheses can stop the generation. That
                // branches exponentially with a batch-1 Core ML predictor, so this
                // host-side port deliberately bounds the active frontier too.
                current = mergeAndPrune(current, width: beamWidth)
                kept = merge(kept)
                if let bestCurrent = current.max(by: { $0.logScore < $1.logScore }) {
                    let completedAboveCurrent = kept.filter {
                        $0.logScore > bestCurrent.logScore
                    }
                    if completedAboveCurrent.count >= beamWidth {
                        current.removeAll(keepingCapacity: true)
                    }
                }
            }
            kept = mergeAndPrune(kept, width: beamWidth)
        }

        guard let best = kept.max(by: { normalizedScore($0) < normalizedScore($1) }) else {
            return TdtHypothesis(decState: decoderState)
        }
        var committedState = try TdtDecoderState(from: best.decoderState)
        committedState.lastToken = best.history.last ?? decoderState.lastToken
        committedState.phraseBoostingState = best.phraseState
        committedState.predictorOutput = best.decoderProjection
        committedState.timeJump = TdtFrameNavigation.calculateFinalTimeJump(
            currentTimeIndices: best.nextFrame,
            effectiveSequenceLength: navigation.effectiveSequenceLength,
            isLastChunk: isLastChunk
        )
        if isLastChunk {
            committedState.finalizeLastChunk()
        }
        decoderState = committedState

        var result = TdtHypothesis(decState: committedState)
        result.score = best.logScore
        result.ySequence = best.outputTokens
        result.timestamps = best.timestamps
        result.tokenDurations = best.durations
        result.tokenConfidences = best.confidences
        result.lastToken = committedState.lastToken
        return result
    }

    private func normalizedScore(_ hypothesis: BeamHypothesis) -> Float {
        // NeMo includes the initial blank/SOS item in y_sequence when it applies
        // final length normalization.
        hypothesis.logScore / Float(hypothesis.history.count + 1)
    }

    private func hypothesisRanksBefore(_ left: BeamHypothesis, _ right: BeamHypothesis) -> Bool {
        if left.logScore != right.logScore { return left.logScore > right.logScore }
        if left.history != right.history {
            return left.history.lexicographicallyPrecedes(right.history)
        }
        if left.nextFrame != right.nextFrame { return left.nextFrame < right.nextFrame }
        return left.phraseState < right.phraseState
    }

    private func mergeAndPrune(
        _ hypotheses: [BeamHypothesis],
        width: Int
    ) -> [BeamHypothesis] {
        Array(merge(hypotheses).sorted(by: hypothesisRanksBefore).prefix(width))
    }

    private func merge(_ hypotheses: [BeamHypothesis]) -> [BeamHypothesis] {
        var merged: [HypothesisKey: BeamHypothesis] = [:]
        merged.reserveCapacity(hypotheses.count)
        for hypothesis in hypotheses {
            if var existing = merged[hypothesis.key] {
                let combinedScore = TdtBeamSearchMath.logAddExp(
                    existing.logScore,
                    hypothesis.logScore
                )
                if hypothesis.logScore > existing.logScore {
                    existing = hypothesis
                }
                existing.logScore = combinedScore
                merged[hypothesis.key] = existing
            } else {
                merged[hypothesis.key] = hypothesis
            }
        }
        return Array(merged.values)
    }
}
