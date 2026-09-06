// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Graph portions Copyright 2023 Xiaomi Corp. (Wei Kang). Apache-2.0.
// Derived from NVIDIA NeMo Speech VarBPEExtension, Copyright NVIDIA Corporation (Apache-2.0).
import Foundation

/// Enumerates case and merged-token arcs independently of a model's greedy text encoder.
public struct VariativeBPEVocabulary: Sendable {
    public enum Segmentation: Sendable {
        case unicodeScalars
        /// Retains the original FluidAudio Parakeet tokenizer's segmentation behavior.
        case graphemes
    }
    private let canonicalIDByTokenID: [Int: Int]
    private let alternativesByCanonicalID: [Int: [Int]]
    private let canonicalSplitByTokenID: [Int: [Int]]
    private let tokenIDsByCanonicalSplit: [[Int]: [Int]]
    private let maximumTokenLength: Int

    public init(vocabulary: [Int: String], segmentation: Segmentation = .unicodeScalars) {
        var byPiece: [String: Int] = [:]
        for (id, piece) in vocabulary where id >= 0 {
            byPiece[piece] = min(id, byPiece[piece] ?? id)
        }
        func segments(_ text: String) -> [String] {
            switch segmentation {
            case .unicodeScalars: return text.unicodeScalars.map(String.init)
            case .graphemes: return text.map(String.init)
            }
        }
        var canonicalIDs: [Int: Int] = [:]
        for (piece, id) in byPiece {
            let lowercase = piece.lowercased()
            canonicalIDs[id] = lowercase != piece ? (byPiece[lowercase] ?? id) : id
        }
        canonicalIDByTokenID = canonicalIDs

        var alternatives: [Int: [Int]] = [:]
        for id in canonicalIDs.keys.sorted() {
            guard let canonicalID = canonicalIDs[id] else { continue }
            alternatives[canonicalID, default: []].append(id)
        }
        for canonicalID in Array(alternatives.keys) {
            alternatives[canonicalID]?.sort { left, right in
                if left == canonicalID { return true }
                if right == canonicalID { return false }
                return left < right
            }
        }
        alternativesByCanonicalID = alternatives

        var splits: [Int: [Int]] = [:]
        var bySplit: [[Int]: [Int]] = [:]
        var maxLength = 1
        for (piece, id) in byPiece {
            let canonicalID = canonicalIDs[id] ?? id
            let split: [Int]
            if segments(piece).count == 1 || (piece.hasPrefix("<") && piece.hasSuffix(">")) {
                split = [canonicalID]
            } else {
                let candidate = segments(piece).compactMap { character -> Int? in
                    guard let characterID = byPiece[String(character)] else { return nil }
                    return canonicalIDs[characterID] ?? characterID
                }
                split = candidate.count == segments(piece).count ? candidate : [canonicalID]
            }
            splits[id] = split
            bySplit[split, default: []].append(id)
            maxLength = max(maxLength, split.count)
        }
        for split in Array(bySplit.keys) {
            bySplit[split]?.sort()
        }
        canonicalSplitByTokenID = splits
        tokenIDsByCanonicalSplit = bySplit
        maximumTokenLength = maxLength
    }
    public func representation(for greedyIDs: [Int]) -> VariativeBPERepresentation? {
        guard !greedyIDs.isEmpty else { return nil }
        let canonicalLengths = greedyIDs.compactMap { canonicalSplitByTokenID[$0]?.count }
        guard canonicalLengths.count == greedyIDs.count else { return nil }
        let canonicalIDs = greedyIDs.flatMap { canonicalSplitByTokenID[$0] ?? [] }
        guard canonicalIDs.count == canonicalLengths.reduce(0, +) else { return nil }

        var groups = [[VariativeBPERepresentation.TokenWithLength]]()
        groups.reserveCapacity(canonicalIDs.count)
        for index in canonicalIDs.indices {
            let canonicalID = canonicalIDs[index]
            var group = (alternativesByCanonicalID[canonicalID] ?? [canonicalID]).map {
                VariativeBPERepresentation.TokenWithLength(token: $0, length: 1)
            }

            let earliestStart = max(0, index - maximumTokenLength)
            if earliestStart < index {
                for start in earliestStart..<index {
                    let split = Array(canonicalIDs[start...index])
                    for token in tokenIDsByCanonicalSplit[split] ?? []
                    where !group.contains(where: { $0.token == token }) {
                        group.append(
                            VariativeBPERepresentation.TokenWithLength(
                                token: token,
                                length: index - start + 1
                            )
                        )
                    }
                }
            }
            guard !group.isEmpty else { return nil }
            groups.append(group)
        }
        return VariativeBPERepresentation(
            canonicalLengths: canonicalLengths,
            tokenGroups: groups
        )
    }
}
