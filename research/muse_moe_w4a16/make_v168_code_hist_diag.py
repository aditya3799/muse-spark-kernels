"""Build a prepare-only NVFP4 code-distribution diagnostic from public v167.

The timed kernel is byte-for-byte the pinned SC6/MFIN source.  The appended
prepare wrapper scans only the 32 local experts and reports exact nibble
frequencies plus simple palette-code lower bounds.  This determines whether a
lossless common-code/escape representation can remove enough HBM traffic to
justify a CUDA implementation.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v167_keell0renz_890809c.py"
DST = ROOT / "submission_v168_code_hist_diag.py"

WRAPPER = r'''


# ---------------------------------------------------------------------------
# v168 prepare-only diagnostic: exact NVFP4 nibble distribution for the local
# experts.  The timed path and every compiled CUDA source remain unchanged.
_v168_prior_prepare = prepare


def _v168_hist_codes(x, chunk_bytes=16 * 1024 * 1024):
    hist = torch.zeros(16, dtype=torch.int64, device=x.device)
    flat = x.reshape(-1)
    for off in range(0, flat.numel(), chunk_bytes):
        u = flat[off:off + chunk_bytes]
        hist += torch.bincount((u & 15).to(torch.int64), minlength=16)
        hist += torch.bincount((u >> 4).to(torch.int64), minlength=16)
    return hist.cpu().tolist()


def _v168_report(name, hist):
    import math
    total = sum(hist)
    order = sorted(range(16), key=lambda c: (-hist[c], c))
    probs = [hist[c] / total for c in order]
    entropy = -sum(p * math.log2(p) for p in probs if p)
    cov = {k: sum(probs[:k]) for k in (1, 2, 3, 4, 7, 8)}
    # Idealized fixed-prefix lower bounds before escape offsets/alignment.
    b2e = 2.0 + 4.0 * (1.0 - cov[3])  # three common codes + escape
    b3e = 3.0 + 4.0 * (1.0 - cov[7])  # seven common codes + escape
    print(
        f"[v168-codehist] {name} total={total} entropy={entropy:.4f} "
        f"order={order} counts={hist} "
        f"cov1={cov[1]:.4%} cov2={cov[2]:.4%} cov3={cov[3]:.4%} "
        f"cov4={cov[4]:.4%} cov7={cov[7]:.4%} cov8={cov[8]:.4%} "
        f"ideal_2bit_escape={b2e:.4f} ideal_3bit_escape={b3e:.4f}",
        flush=True,
    )


def prepare(weights):
    state = _v168_prior_prepare(weights)
    lo, hi = LOCAL_START, LOCAL_START + LOCAL_EXPERTS
    h13 = _v168_hist_codes(weights["w13"][lo:hi])
    h2 = _v168_hist_codes(weights["w2"][lo:hi])
    _v168_report("w13", h13)
    _v168_report("w2", h2)
    _v168_report("combined", [a + b for a, b in zip(h13, h2)])
    torch.cuda.synchronize()
    return state
'''


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    if not source.startswith("# okc: problem=muse_moe_w4a16\n"):
        raise RuntimeError("unexpected source header")
    if "_v168_prior_prepare" in source:
        raise RuntimeError("source already contains v168 wrapper")
    DST.write_text(source + WRAPPER, encoding="utf-8")
    print(f"wrote {DST.name} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
