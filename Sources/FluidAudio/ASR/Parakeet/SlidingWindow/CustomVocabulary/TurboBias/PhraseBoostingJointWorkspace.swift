@preconcurrency import CoreML
import Foundation

/// Reusable tensors for the optional full-vocabulary joint prediction.
///
/// TurboBias calls this only after the optimized joint has selected a non-blank
/// category. Keeping all inputs and output storage alive for the decode avoids a
/// per-token Core ML feature-provider/tensor allocation cycle.
internal final class PhraseBoostingJointWorkspace {
    private let encoderStep: MLMultiArray
    private let decoderStep: MLMultiArray
    private let logitsBacking: MLMultiArray
    private let inputProvider: MLFeatureProvider
    private let predictionOptions: MLPredictionOptions
    private let decoderHiddenSize: Int
    private let outputSize: Int

    init(encoderHiddenSize: Int, decoderHiddenSize: Int, outputSize: Int) throws {
        self.decoderHiddenSize = decoderHiddenSize
        self.outputSize = outputSize
        encoderStep = try MLMultiArray(
            shape: [1, 1, NSNumber(value: encoderHiddenSize)], dataType: .float32)
        decoderStep = try MLMultiArray(
            shape: [1, 1, NSNumber(value: decoderHiddenSize)], dataType: .float32)
        logitsBacking = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: outputSize)], dataType: .float32)
        inputProvider = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderStep),
            "decoder_outputs": MLFeatureValue(multiArray: decoderStep),
        ])
        predictionOptions = MLPredictionOptions()
        predictionOptions.outputBackings = ["logits": logitsBacking]
    }

    func scores(
        encoderFrames: EncoderFrameView,
        timeIndex: Int,
        preparedDecoderStep: MLMultiArray,
        model: MLModel
    ) throws -> [Float] {
        let encoderStrides = encoderStep.strides.map(\.intValue)
        let encoderPointer = encoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: encoderStep.count)
        try encoderFrames.copyFrame(
            at: timeIndex,
            into: encoderPointer,
            destinationStride: encoderStrides[2]
        )

        let sourceShape = preparedDecoderStep.shape.map(\.intValue)
        guard preparedDecoderStep.dataType == .float32,
            sourceShape == [1, decoderHiddenSize, 1]
        else {
            throw ASRError.processingFailed(
                "Prepared phrase-boosting decoder input has an unexpected shape")
        }
        let sourceStride = preparedDecoderStep.strides[1].intValue
        let destinationStride = decoderStep.strides[2].intValue
        let source = preparedDecoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: preparedDecoderStep.count)
        let destination = decoderStep.dataPointer.bindMemory(
            to: Float.self, capacity: decoderStep.count)
        for index in 0..<decoderHiddenSize {
            destination[index * destinationStride] = source[index * sourceStride]
        }

        encoderStep.prefetchToNeuralEngine()
        decoderStep.prefetchToNeuralEngine()
        let output = try model.prediction(from: inputProvider, options: predictionOptions)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
            logits.dataType == .float32,
            logits.count == outputSize
        else {
            throw ASRError.processingFailed(
                "Phrase-boosting joint returned an unexpected logits tensor")
        }

        let stride = logits.strides.last?.intValue ?? 1
        let pointer = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        return (0..<outputSize).map { pointer[$0 * stride] }
    }
}
