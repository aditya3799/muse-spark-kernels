# Routed NVFP4 experts

Published standing on 20 November 2026: **110 / 1,422**, **2.4389x** SGLang. Official job: `5f5a8c8b72b24ee4`; recorded correctness: **5/5**. This is a kernel-track result, not full-model serving throughput.

## Implementation and evidence

GPU route construction, packed NVFP4 expert projections, SwiGLU, down projection, per-expert RMS normalization, and weighted combination. Research includes WGMMA/warp schedules, K-layout permutations, producer waits, and finalization.

**Source provenance limitation:** the current workspace file was changed after the best scored job. `baseline_snapshot.py` is the preserved backup taken before subsequent experiments; `working_snapshot.py` is the later working source. Exact byte identity between either snapshot and the leaderboard-winning payload has not been established. The score below belongs to the official job, not a verified rerun of these snapshots.

- [Selected source snapshot](baseline_snapshot.py)
- [Official result summary](../results/muse_moe_w4a16.job.json)
- [Current standings and fetch timestamps](../results/standings.json)
- [Source hashes and provenance](../results/source_provenance.json)
- [Historical research](../research/muse_moe_w4a16/)
- [Official problem](https://openkernelchallenge.dev/problems/muse_moe_w4a16)

Use the official `okc init muse_moe_w4a16` scaffolding and the contest's Linux/Hopper environment for runtime validation. Read the [kernel-track reproduction notes](../README.md#source-map-and-reproduction) first. Source headers retain upstream attribution; see [NOTICE](../../NOTICE.md).
