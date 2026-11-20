# v164 WGS-only polynomial SiLU

## Status

Local source, numerical, and exact-SM90a static gates pass. The authorized H200
diagnostic passed correctness but failed the predeclared performance gate, so
v164 is rejected. No scored run was consumed. The diagnostic remains
default-off and the official/scored path is frozen v139.

## Motivation

A fresh source-attributed disassembly of v139 found 233 static MUFU sites per
WGS kernel template. Of those, 216 are in the phase-A SwiGLU epilogue: the
exact `exp` plus reciprocal sequence. This is the largest remaining
special-function hotspot after the WGMMA/dequant/scheduling branches were
closed.

v164 changes only that WGS phase-A operation. M1/M8/M16 continue to use their
existing exact SiLU paths. The candidate reuses the polynomial already studied
in the older v77 branch:

```cpp
const float z = x * x;
const float p = fmaf(fmaf(2.82057459e-4f, z, -1.16906086e-2f), z,
                     2.34261343e-1f);
return x * __saturatef(fmaf(x, p, .5f));
```

The diagnostic is enabled only by `-DOKC_WGSPOLY=1`; the macro defaults to 0.

## Local correctness evidence

- `audit_v164_polysilu.py` passes.
- The edit occurs exactly once at the WGS phase-A epilogue.
- Mixed uniform-grid/Gaussian scalar relative L2: **0.267579%**.
- Scalar maximum absolute error over that audit: **0.053391**.
- The embedded H200 diagnostic will compare candidate versus v139 on seven
  seeds each at M32/M128/M256 and reject relative L2 above 2.0% or a maximum
  per-row relative error above 6.0%.
- Earlier end-to-end v77 measurements of this same polynomial were 0.96-0.99%
  combined-output error on normal routing and 0.72-0.73% on skew routing, with
  worst row below 1.16%.

## Exact SM90a static evidence

| Metric | v139/control | WGS polynomial |
|---|---:|---:|
| Registers | 96 | 96 |
| Stack | 48 B | 48 B |
| Spill stores | 44 B | 44 B |
| Spill loads | 104 B | 104 B |
| WGMMA serialization | No | No |
| HGMMA sites | 1,024 | 1,024 |
| MUFU sites | 466 | 34 |
| FFMA sites | 246 | 894 |
| IMAD sites | 5,966 | 6,168 |

The macro-off control disassembly is byte-identical to the freshly compiled
frozen v139 disassembly. Both SASS files hash to
`F492D1575712DE4321DB409BDB298A882FA90754CD755973570ADE5B48D003CD`.

The candidate therefore removes all 432 targeted MUFU sites across the two
kernel templates without changing registers, stack, spills, HGMMA inventory,
or WGMMA scheduling.

## Remote diagnostic verdict

Authorized check `a85addbaf44340c3` ran 15 alternating cold-L2 graph replays
for each of seven seeds at M32/M128/M256. It passed all five official
correctness cases with no flags or rejections. All 21 embedded candidate versus
v139 output comparisons also passed:

- maximum global candidate/control relative L2: **0.9994%**;
- maximum per-row candidate/control relative error: **1.1305%**.

The timing result was:

| Shape | v139 median | polynomial median | paired median ratio | wins |
|---|---:|---:|---:|---:|
| M32 | 134.3 us | 132.9 us | 0.9976x | 4/7 |
| M128 | 200.8 us | 200.6 us | 0.9994x | 4/7 |
| M256 | 215.9 us | 215.4 us | 0.9999x | 5/7 |

The mean over all 21 paired ratios was **0.9989x**. Promotion required all of
the following:

- M32 median candidate/control ratio <= 0.995;
- M128 and M256 median ratios <= 1.002;
- mean of all 21 paired ratios <= 0.9975;
- embedded output-drift limits pass.

The candidate failed the M32 median and 21-case mean requirements, and the
embedded harness reported `PROMOTE=False`. The saved aggregate result of
5701.33 us / 2.35884x is the macro-off v139 path and is not candidate evidence.
Reject v164 without promotion or a scored run. The result also closes exact
SiLU special-function latency as a dominant WGS bottleneck: removing 432 MUFU
sites was numerically safe but produced only a noise-scale timing change.

## Integrity

- v164 diagnostic SHA256:
  `3A941F1244B27D2CFAB9EFBF5AEE6B91A65F397046AAA5B4FC7DACD606CFB9C2`
- frozen v139 SHA256:
  `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`
- live `submission.py` SHA256:
  `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`
- remaining quota: one check and five scored runs.

## Artifacts

- `submission_v164_wgs_polysilu_diag.py`
- `make_v164_wgs_polysilu_diag.py`
- `audit_v164_polysilu.py`
- `compile_v164_polysilu_static.py`
- `analyze_v164_v139_sass.py`
- `profiles/submission_v164_wgs_polysilu_diag.a85addbaf44340c3.log`
- `profiles/submission_v164_wgs_polysilu_diag.a85addbaf44340c3.json`
- `profiles/submission_v164_wgs_polysilu_diag.a85addbaf44340c3.txt`
