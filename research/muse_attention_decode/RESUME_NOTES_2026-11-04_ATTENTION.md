# Attention decode all-out pass — 2026-11-04

## 2026-11-05 continuation — new scored winner

- Active file: `submission.py`
- SHA256: `3e6d676275706a76c624b27a970583e6157a8cc536c4843d72fdc40b986172a2`
- Scored job: `fcf3d95f5f2046bc`
- Correctness/static checks: 16/16, no flags
- Score: 1398.2995417032162 us
- Speedup: 2.6934858288034706x (95% interval 2.6530835413714184–2.745654655657574x)
- Improvement over job `07e5d76c9c114a4b`: 1440.488284857772 -> 1398.2995417032162 us
  (-42.1887431545558 us, -2.93%); speedup 2.6145995351651674x -> 2.6934858288034706x (+3.02%).
- Live rank immediately after promotion: attention 107/1351; overall endpoint 173/1981.
- Change: enable the exact q8 resident-cooperative uneven per-SM work split (`OKC_W0ENABLE=1`) at the previously
  tuned 1135/2048 share. The source restricts registration to `L == 8`, preserving q1's compile-time fast partition.
- Hot resources are unchanged: q1 flat 80 registers / 0-byte stack; q8 flat 80 registers / 32-byte stack.

Final scored full-shape medians were q1 B16/B32 84.655/160.666 us and q8 B16/B32 97.118/175.490 us.
The winner was first checked off-board (`32a6ff08b6a54812`, 1448.223 us, 16/16) and then reproduced as the scored
winner above. Daily state after promotion: scored 6/20 used, checks 40/40 used.

### Additional exact experiments closed in the continuation

- q1 four-CTA/two-stage NP128 (`fbef333816bc4491`): 1539.447 us; q1 full regressed to 106.968/199.892 us.
- q1 four-CTA/two-stage NP64 compiled at 80 registers / 16-byte stack, but was pruned because the topology itself
  was the NP128 failure and NP64 halves producer warps.
- B32-only page-index prefetch disable (`74514bca392841e3`): 1452.030 us; reject.
- V `.L2::128B` plus `evict_first` policy (`6de650224e144bdc`): illegal instruction on the long q8 test; never use.
- Audited B32-only q8 five-stage pipeline graft (`16fc3958f33f40e9`): 1462.044 us. It improved q8 B32 to
  182.457 us but lost aggregate; retain only as research.
- Five-stage + uneven-share combination check (`879cfe9c7408424d`): 1451.488 us, q8 B32 profile 179.9 us.
  Its scored replication (`73dbfea39a664162`) was 1446.639 us / 2.603482x, so the combination was not promoted.
- All candidates remained exact full-attention implementations. No compressed/runtime-decoded source, harness or
  timing manipulation, host synchronization, sparse/top-k/key skipping, value skipping, or disallowed precision.

### Exhaustive scored W0 matrix (2026-11-05)

The user requested consuming every remaining scored attempt. Fourteen non-refunded exact variants were run after the
1398.300-us promotion; none beat it, so `submission.py` remained unchanged.

| Candidate | Job | Score (us) | Speedup |
|---|---|---:|---:|
| W0 1120 | `37181f8886814ce6` | 1452.347 | 2.593250x |
| W0 1125 | `928648ba8c484e98` | 1466.828 | 2.567649x |
| W0 1130 | `ea46607180dc4e58` | 1451.759 | 2.594301x |
| W0 1138 | `8e2e92c4d491463c` | 1459.672 | 2.580238x |
| W0 1140 | `e6e283f1c55a4e6c` | 1457.962 | 2.583263x |
| W0 1145 | `9b37e3effb774c13` | 1461.481 | 2.577043x |
| W0 1150 | `365b75d46f1a4edd` | 1466.314 | 2.568549x |
| B16=1130, B32=1140 | `6d48c5f1eb344dc3` | 1441.267 | 2.613186x |
| B16=1140, B32=1130 | `8a9b1706c8844d1b` | 1464.814 | 2.571180x |
| B16=1135, B32=1140 | `dc00af4e5af74580` | 1462.685 | 2.574922x |
| B16=1140, B32=1135 | `f89ac28b0d1a4bc0` | 1456.249 | 2.586302x |
| SEGK8 550 | `ac571b2062a842b9` | 1469.187 | 2.563527x |
| SEGK8 650 | `534d2ac4703b4d20` | 1448.432 | 2.600260x |
| W0 + q8 K64/V128 L2 hints | `8597de299efa473b` | 1454.014 | 2.590277x |

The W0 + q8 K128/V128 job `dcc795c3cf91400d` hit a refunded infrastructure error and did not count; the 1138
variant filled that slot. Final quota: scored 20/20, checks 40/40. Final live state: attention 107/1351 (64 attempts),
overall 173/1981, active job `fcf3d95f5f2046bc`, active hash
`3e6d676275706a76c624b27a970583e6157a8cc536c4843d72fdc40b986172a2`.

## Prior scored incumbent (superseded on 2026-11-05)

- Active file: `submission.py`
- SHA256: `1edef4d178cdbee3b256ba3d662738a0f5ba7788c1782abcf012341f8a3f20c9`
- Scored job: `9408bd801dd04353`
- Correctness: 16/16
- Score: 1444.3444114015174 us
- Speedup: 2.6076190486626225x (95% interval 2.5685–2.6571x)
- Live rank after this pass: attention 303; overall kernel board 167.
- `submission.py` was not overwritten because neither new scored candidate beat it.

## Eligibility constraints preserved

- No compressed/encoded source or runtime decoding. This was the prior BF16/decode disqualification reason and was
  explicitly scanned in every new candidate.
- No harness/timing mutation, network/process access, input caching, host synchronization, key skipping, sparse/top-k
  attention, or activation precision below the permitted FP8 query / FP8 KV path.
- Imported CUDA is plain Apache-2.0 source with attribution retained.
- The public keell0renz file enabled `ZH_SCH=1`, described in its own source as a measurement-only piecewise-linear
  exponent path. All candidates in this pass forced `ZH_SCH=0`; that approximate path was never submitted.

## Public bulk/WST/VP2 ablation matrix

All jobs below passed 16/16 and were off-board checks unless marked scored.

| Job | Candidate | Score (us) | Speedup | Result |
|---|---|---:|---:|---|
| `e31d3f98ee8c446b` | public composite, exact WST + exact VP2 | 1476.7 | 2.550x | exact WST bad |
| `8ee8926fae4641bb` | composite + incumbent `.L2::64B` fast-path hint | 1469.5 | 2.563x | L2 port valid, WST still bad |
| `fb13c7a05f654fc3` | bulk/L2 + exact WST only | 1468.3 | 2.565x | WST B16 full ≈123.3 us |
| `1fae061aa96b479b` | bulk/L2 + exact VP2 only | 1450.1 | 2.597x | VP2 helps, public base loses |
| `bdb77ae1a8ff4e78` | bulk/L2 base control | 1461.7 | 2.577x | public base slower than incumbent |

Exact VP2 reduced verify-B1 sliding from the base control's 6.8 us to 6.2 us and improved that public-base aggregate by
about 11.6 us. Exact WST increased verify-B16 full from about 100–101 us to 123–124 us; it is closed.

## Exact direct-E4M3 WST experiment

- Derived a direct correctly-rounded E4M3FN code for `2^y` in the log2 domain.
- Exhaustive CPU validation covered all 50,274 relevant FP16 bit patterns through `y <= 8.75`: zero mismatches against
  `exp2 -> E4M3FN` round-to-nearest-even, including the exact `y=-10` zero/min-subnormal tie.
- Direct1 kept 80 registers / zero stack and cut static `MUFU.EX2` occurrences from 147 to 19, but force-inlining grew the
  WST kernel from 4,752 to 11,408 instructions. Job `e2c64f2366b74a27`: 2072.5 us; WST B16 full 1167.6 us.
- Direct4 grouped calls: 80 registers, 16-byte stack, 5,032 instructions, 19 `MUFU.EX2`. Job
  `cbdeeff003614d2e`: 2082.3 us; WST B16 full 1083.9 us.
- Correctness passed, but branch/call divergence destroys performance. Close the entire direct-threshold WST family.

## Incumbent + exact VP2

Two packaging variants preserved the incumbent for eleven shapes and routed only scored verify-B1 sliding to VP2.

| Job | Packaging | Score (us) | Speedup | Board |
|---|---|---:|---:|---|
| `dc86d2525d4448a6` | two extensions, check | 1456.8 | 2.585x | off-board |
| `a70ef28364444466` | two extensions, scored | 1459.9 | 2.580x | lost to incumbent |
| `c2038054650b4b51` | one integrated extension, check | 1443.6 | 2.609x | off-board; tiny win inside CI |
| `984367f898df4a04` | one integrated extension, scored | 1458.3 | 2.583x | lost to incumbent |

The integrated binary retained all incumbent resource counts and 16 `LDGSTS.E.BYPASS.LTC64B.128` instructions; VP2
used 99 registers and zero stack/local memory. The favorable 1443.6-us check did not reproduce when scored, so do not
promote or resubmit comment-only variants to chase judge noise.

## Files and local build note

- Strongest experimental source: `submission_incumbent_vp2_integrated.py` (valid but not promoted).
- Mechanical composers: `compose_incumbent_vp2.py`, `compose_incumbent_vp2_integrated.py`.
- C: fell to about 127 MB free during compilation. No user data was deleted. Local builds were redirected to
  `.local_build_temp/` and `.torch_extensions/` under this workspace on D:.
- Hopper/SGLang research supported the existing direction: TMA/async-copy and split-KV reduction are the relevant
  levers; generic FA3/FlashMLA replacement was not a fit for this fused gated, page-size-1, head-dim-64 workload.

## Closed directions from this pass

- Public composite with measurement-only approximate exponent: eligibility risk; never submit.
- Public composite with exact exponent: valid but slower.
- Exact WST: slower by ~23 us on its only routed scored shape.
- Direct exact E4M3 WST: mathematically correct, catastrophically slow on GPU.
- Public bulk-reduction base: slower than incumbent.
- Exact VP2 on public base: real target-shape win, aggregate still behind incumbent.
- VP2 grafts onto incumbent: valid; favorable check did not reproduce in scored runs. Retain incumbent.
