# Gated attention decode

Published standing on 20 November 2026: **126 / 1,379**, **2.6935x** SGLang. Official job: `fcf3d95f5f2046bc`; recorded correctness: **16/16**. This is a kernel-track result, not full-model serving throughput.

## Implementation and evidence

Fused Q/K normalization, RoPE where needed, FP8 cache update, online-softmax attention, RMS normalization, and gating. The final documented change enabled q_len=8 cooperative work distribution; rejected occupancy and pipeline combinations remain in the archive.

The selected source matches the documented final attention incumbent hash.

- [Selected source snapshot](submission.py)
- [Official result summary](../results/muse_attention_decode.job.json)
- [Current standings and fetch timestamps](../results/standings.json)
- [Source hashes and provenance](../results/source_provenance.json)
- [Historical research](../research/muse_attention_decode/)
- [Official problem](https://openkernelchallenge.dev/problems/muse_attention_decode)

Use the official `okc init muse_attention_decode` scaffolding and the contest's Linux/Hopper environment for runtime validation. Read the [kernel-track reproduction notes](../README.md#source-map-and-reproduction) first. Source headers retain upstream attribution; see [NOTICE](../../NOTICE.md).
