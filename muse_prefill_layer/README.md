# Whole prefill layer

Published standing on 20 November 2026: **54 / 895**, **1.9360x** SGLang. Official job: `399cbda410c440ee`; recorded correctness: **5/5**. This is a kernel-track result, not full-model serving throughput.

## Implementation and evidence

Dense GEMMs and FA3 attention combined with fused Triton boundaries and Hopper grouped NVFP4 expert GEMMs. Persistent scheduling, tile-selective exact K layouts, and exact scale sharing were investigated. Lower-precision storage experiments are historical and do not supply this leaderboard result.

Selected: the explicit `submission_scored_399cbda410c440ee.py` snapshot, hash `7a5aedf8b6def500d72463a5a12eea7f9bb07c49a8977b400ccb27739d23676e`.

- [Selected source snapshot](submission.py)
- [Official result summary](../results/muse_prefill_layer.job.json)
- [Current standings and fetch timestamps](../results/standings.json)
- [Source hashes and provenance](../results/source_provenance.json)
- [Historical research](../research/muse_prefill_layer/)
- [Official problem](https://openkernelchallenge.dev/problems/muse_prefill_layer)

Use the official `okc init muse_prefill_layer` scaffolding and the contest's Linux/Hopper environment for runtime validation. Read the [kernel-track reproduction notes](../README.md#source-map-and-reproduction) first. Source headers retain upstream attribution; see [NOTICE](../../NOTICE.md).
