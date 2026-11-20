# Whole decoder layer

Published standing on 20 November 2026: **194 / 1,296**, **2.2684x** SGLang. Official job: `c3d4cf6955bf4cdc`; recorded correctness: **13/13**. This is a kernel-track result, not full-model serving throughput.

## Implementation and evidence

Full graph-capturable layer composition: dense projections, attention, routing, experts, and residual/norm boundaries. S108 integrates the attention and exact small-token projection changes. Whole-layer comparisons rejected several component changes that did not transfer to the combined critical path.

Selected: `submission_s108_att_cgemv.py`, associated with the scored S108 job by the saved submission receipt and subsequent comparison scripts. `working_snapshot.py` preserves the later S128 work.

- [Selected source snapshot](submission.py)
- [Official result summary](../results/muse_decode_layer.job.json)
- [Current standings and fetch timestamps](../results/standings.json)
- [Source hashes and provenance](../results/source_provenance.json)
- [Historical research](../research/muse_decode_layer/)
- [Official problem](https://openkernelchallenge.dev/problems/muse_decode_layer)

Use the official `okc init muse_decode_layer` scaffolding and the contest's Linux/Hopper environment for runtime validation. Read the [kernel-track reproduction notes](../README.md#source-map-and-reproduction) first. Source headers retain upstream attribution; see [NOTICE](../../NOTICE.md).
