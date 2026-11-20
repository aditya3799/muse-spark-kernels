# Dense projections

Published standing on 20 November 2026: **72 / 1,740**, **1.5967x** SGLang. Official job: `ac25b659640245d1`; recorded correctness: **9/9**. This is a kernel-track result, not full-model serving throughput.

## Implementation and evidence

Skinny BF16 GEMV/GEMM, lossless coefficient packing, and guarded route composition for QKV/gate, output, pooler/unpooler, and the FP32 router. The development cycle emphasized coefficient recovery, probe rows, FP64 comparisons, resource checks, and paired cold-cache timing.

Selected: `submission_plain_full_guard_v221.py` from the workspace, with documented SHA-256 `bec0d91828a3fc5b2311ee5963ad8ae83920b9fe660c0c0a333b3c32e1fb4dd9`. The original top-level working file is preserved separately.

- [Selected source snapshot](submission.py)
- [Official result summary](../results/muse_bf16_proj.job.json)
- [Current standings and fetch timestamps](../results/standings.json)
- [Source hashes and provenance](../results/source_provenance.json)
- [Historical research](../research/muse_bf16_proj/)
- [Official problem](https://openkernelchallenge.dev/problems/muse_bf16_proj)

Use the official `okc init muse_bf16_proj` scaffolding and the contest's Linux/Hopper environment for runtime validation. Read the [kernel-track reproduction notes](../README.md#source-map-and-reproduction) first. Source headers retain upstream attribution; see [NOTICE](../../NOTICE.md).
