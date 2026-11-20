# v157 M32 segmented final-pass scout

Status: H200 correctness passed; both segmented variants are slower and rejected; no score consumed.

## Candidate

`submission_v157_wgs_finalseg_diag.py` adds a default-off `OKC_WGSFSEG` switch with diagnostic variants for two
and four CTAs per M32 token.  The protected v139 production defines do not enable it.  At M32, each active CTA owns
a disjoint column segment, computes eight route sum-of-squares partials, and stores them in the launch's otherwise
unused double-buffered `ss` copy.  A second grid barrier makes those partials visible; each active CTA then forms
the eight normalization factors and writes its disjoint BF16 output segment.  M != 32 and grids with fewer than
`32 * OKC_WGSFSEG` CTAs take the original whole-row final pass.

No allocation or scratch-size change is required.  FSEG2 uses 64 active CTAs with 512 output threads per CTA;
FSEG4 uses 128 active CTAs with 256 output threads per CTA.  The largest partial index is 1,023, within the existing
3,072-float `ss` copy.

## Barrier contract

The existing first grid barrier still uses its generation-parity counter.  In candidate builds, its last arriver
resets `ctr[0]`, performs a device fence, then increments the launch generation.  Every other CTA waits on that
generation rather than only the arrival count, so no CTA can reach the second barrier before the reset is visible.
All CTAs (including inactive ones) then fence, increment `ctr[0]` once, and wait for `gridDim.x`.  `ctr[0]` may remain
at the grid size until the next launch: that launch's first-barrier last arriver resets it before publishing the next
generation.  This also handles alternating M32 and fallback launches.

`audit_v157_finalseg.py` proves exact token/segment ownership, exact 0..4095 column coverage for FSEG2/FSEG4,
partial-scratch bounds, required source ordering, macro-off production dispatch, and alternating barrier epochs.

## Static evidence

`compile_v157_finalseg_static.py` compiled frozen v139, default-off v157, FSEG2, and FSEG4 for CUDA `sm_90a`.
The default-off v157 SASS is byte-for-byte identical to v139.  Both candidates preserve 96 registers, the same
eight HGMMA widths and all 1,024 HGMMA sites, with no serialization warning:

| build | stack | spill stores / loads | registers |
|---|---:|---:|---:|
| v139 / default-off v157 | 48 B | 44 / 104 B | 96 |
| FSEG2 | 48 B | 4 / 12 B | 96 |
| FSEG4 | 48 B | 4 / 12 B | 96 |

The final guarded rebuild passed after adding the small-grid fallback.

## Frozen local artifacts

- `submission_v157_wgs_finalseg_diag.py`:
  `587AA8F4540764D0D7F6F0203887AF92D267D2877BF7E25193CAC1FDD32A7E6E`
- `compile_v157_finalseg_static.py`:
  `5E717EB6ADA755DCC732761624ADBE74262697EA58F11C657B141D0989BE6F7E`
- `audit_v157_finalseg.py`:
  `799ADE8D7725FABA7796599CDAA098BA9D1D4E546A47B47E48D0B2BFE4EF75A2`

Protected files remain unchanged:

- v139: `A24B8374AAD49FBAA5E0CDEFFB7F1FA64960D8321F343CF8B88BD97DFDD20C09`
- `submission.py`: `57D900122BFEE50C2CDD01D6B4D884CEC4ADE15E48F11F58BD6E056B2B870C38`

## H200 verdict

Authorized check job `6bc4bb3bfaea45e1` completed successfully on H200.  All five official correctness cases
passed with no flags or rejections.  The diagnostic harness also completed both independent invocations across
M32/M33/M128/M256 and five seeds per invocation, so both candidates stayed within its relative-L2 gate.

The performance gate failed in both invocations:

| sweep | variant | M32 ratio | M33 | M128 | M256 |
|---:|---|---:|---:|---:|---:|
| 1 | FSEG2 | 1.0076x | 1.0044x | 1.0016x | 1.0042x |
| 1 | FSEG4 | 1.0113x | 1.0102x | 1.0018x | 1.0052x |
| 2 | FSEG2 | 1.0111x | 1.0049x | 1.0046x | 1.0048x |
| 2 | FSEG4 | 1.0101x | 1.0057x | 1.0002x | 0.9991x |

FSEG2 therefore regressed M32 by 0.76-1.11%, and FSEG4 by 1.01-1.13%, far from the <=0.985 promotion gate.
The extra grid barrier and partial-scratch round trip cost more than the parallel final pass saves.  Reject v157;
do not score or promote either variant.  The macro-off check path measured 5,645.78 us / 2.38204x, but that aggregate
is not candidate evidence because `kernel_fn` deliberately remains on v139.  After the check, the service reports
5 checks and 5 scored runs remaining for this problem today.
