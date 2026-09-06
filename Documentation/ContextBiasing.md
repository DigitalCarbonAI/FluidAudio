# Shared phrase graph

`ContextBiasing` is a Foundation-only product containing the TurboBias graph, state-potential
scoring, failure links, phrase matches and variative BPE arcs. It depends on neither FluidAudio
nor an acoustic runtime. `FluidAudio` depends on it; other decoders can import it independently.

The tokenizer supplies greedy token IDs and optional `VariativeBPERepresentation` values.
`VariativeBPEVocabulary` enumerates casing and merge alternatives but does not encode text.
Its default segmentation uses Unicode scalars, matching Python character iteration. The
Parakeet wrapper explicitly retains the prior Swift grapheme behavior and its existing greedy
piece encoder, blank boundary, joint probability checks, canonical formatting and pruning.

Decoder policy remains outside the product: TDT preserves base blank/duration selection;
an attention decoder must handle EOS separately. Fusion alpha and vocabulary-size validation
also belong to the caller. There is no EOS token, Core ML model, MLX tensor, global cache,
network access or mutable decoding session in the shared graph.

`candidateTokens(from:)` returns the union of explicit arcs on a state's failure chain.
Every other token shares `transition(from:token:-1).score`; callers can initialize a graph
score row once and fill candidates without reading acoustic logits on the CPU.

The extraction retains the existing nonuniform, case-insensitive TurboBias implementation.
It does not claim to implement every NeMo option (per-phrase alphas, uniform weighting,
beam search or CUDA storage). Source reference: NVIDIA-NeMo/NeMo commit
`ca4daa1470f6c01068c4e6a9a73b19b9a91dc366`, `context_graph_universal.py`,
`boosting_graph_batched.py`, and `sentencepiece_tokenizer.py`.

Verification:

```sh
swift test --disable-keychain --filter 'PhraseBoostingContextTests|PhraseBoostingGraphTests'
```

The existing Parakeet suite covers selection, skip proofs, blank preservation, acoustic
probabilities, casing, split variants, prefix rollback and completion. The new graph suite
covers row sparsity, invalid imported representations, cancellation and Unicode policy.
The OrcaScribe issue #107 companion change additionally checks NeMo-generated golden paths,
including Cohere's full checkpoint vocabulary, and real INT5/INT8 audio qualification.
