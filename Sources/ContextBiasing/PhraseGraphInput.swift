// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Graph portions Copyright 2023 Xiaomi Corp. (Wei Kang). Apache-2.0.
// Derived from NVIDIA NeMo Speech, Copyright NVIDIA Corporation (Apache-2.0).
import Foundation

public enum PhraseGraphError: Error, LocalizedError, Equatable {
    case invalidConfiguration
    case emptyPhrases
    case invalidRepresentation

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration: return "Phrase graph scoring parameters are invalid."
        case .emptyPhrases: return "The phrase graph has no supported phrases."
        case .invalidRepresentation: return "A phrase token representation is invalid."
        }
    }
}

/// Scoring parameters shared by transducer and attention-decoder phrase graphs.
/// Fusion strength and EOS/blank policies belong to the decoder, not this graph.
public struct PhraseGraphConfiguration: Sendable, Equatable {
    public let contextScore: Float
    public let depthScaling: Float
    public let unknownScore: Float
    public let variativeScoringTemperature: Float
    public let penalizeSubsplits: Bool

    public init(
        contextScore: Float = 1, depthScaling: Float = 2, unknownScore: Float = 0,
        variativeScoringTemperature: Float = 10, penalizeSubsplits: Bool = true
    ) {
        self.contextScore = contextScore
        self.depthScaling = depthScaling
        self.unknownScore = unknownScore
        self.variativeScoringTemperature = variativeScoringTemperature
        self.penalizeSubsplits = penalizeSubsplits
    }

    func validate() throws {
        guard contextScore.isFinite, contextScore >= 0, depthScaling.isFinite, depthScaling >= 1,
            unknownScore.isFinite, unknownScore >= 0,
            variativeScoringTemperature.isFinite, variativeScoringTemperature >= 0
        else { throw PhraseGraphError.invalidConfiguration }
    }
}

/// A tokenizer-specific phrase. Greedy token IDs define the score boundaries.
public struct PhraseGraphInput: Sendable {
    public let tokens: [Int]
    public let variative: VariativeBPERepresentation?

    public init(tokens: [Int], variative: VariativeBPERepresentation? = nil) {
        self.tokens = tokens
        self.variative = variative
    }

    func validate() throws {
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 }) else {
            throw PhraseGraphError.invalidRepresentation
        }
        guard let variative else { return }
        try variative.validate()
    }
}

/// Alternate token segmentations converging on one canonical character backbone.
public struct VariativeBPERepresentation: Sendable {
    public struct TokenWithLength: Sendable {
        public let token: Int
        public let length: Int

        public init(token: Int, length: Int) {
            self.token = token
            self.length = length
        }
    }

    public let canonicalLengths: [Int]
    public let tokenGroups: [[TokenWithLength]]

    public init(canonicalLengths: [Int], tokenGroups: [[TokenWithLength]]) {
        self.canonicalLengths = canonicalLengths
        self.tokenGroups = tokenGroups
    }

    func validate() throws {
        guard !canonicalLengths.isEmpty, canonicalLengths.allSatisfy({ $0 > 0 }) else {
            throw PhraseGraphError.invalidRepresentation
        }
        var total = 0
        for length in canonicalLengths {
            let result = total.addingReportingOverflow(length)
            guard !result.overflow else { throw PhraseGraphError.invalidRepresentation }
            total = result.partialValue
        }
        guard total == tokenGroups.count else { throw PhraseGraphError.invalidRepresentation }
        for (index, group) in tokenGroups.enumerated() {
            guard group.first?.length == 1,
                group.allSatisfy({ $0.token >= 0 && $0.length > 0 && $0.length <= index + 1 })
            else { throw PhraseGraphError.invalidRepresentation }
        }
    }
}
