"""Mechanically derive the v160 diagnostic payload from the frozen v150 harness."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "submission_v150_wgs_compute_diag.py"
TARGET = ROOT / "submission_v160_wgs_epinorm_diag.py"
EXPECTED = "26ACE8185AB010EAD1157AA21D900E82379FD34AC0497A70D1693537D0A66444"


def once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)


def main() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper() == EXPECTED
    text = SOURCE.read_text()
    text = once(text, '"""Routed experts (NVFP4 W4A16 MoE) as one persistent CUDA kernel plus two tiny ones.',
                '"""v160 diagnostic: losslessly paired phase-B norm weights on the frozen v139 kernel.')
    text = once(text, """#ifndef OKC_WGSFINSS
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#endif""", """#ifndef OKC_WGSFINSS
#define OKC_WGSFINSS 1  // WGS: the final pass forms each route's sum of squares (the phase-B epilogue only writes y)
#endif
#ifndef OKC_WGSEPINORM
#define OKC_WGSEPINORM 0  // v160: lossless (g,g+8) bf16 norm pair, one LDG.32 in the phase-B epilogue
#endif""")
    text = once(text,
                "  const uint32_t* lut; const __nv_bfloat16* normw;  // post_norm_w as given (bf16); 1 + w is formed in FP32 here",
                "  const uint32_t* lut; const __nv_bfloat16* normw; const uint32_t* normwp;  // original and losslessly paired bf16 norm weights")
    text = once(text, """    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;""",
                """    const float gsc = p.gsB[e] * TWO62;
#if OKC_WGSEPINORM
    const uint32_t nwbits = __ldg(p.normwp + r16 * 8 + g);
    const float w0 = (1.f + __uint_as_float(nwbits << 16)) * gsc;
    const float w1 = (1.f + __uint_as_float(nwbits & 0xffff0000u)) * gsc;
#else
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#endif""")
    text = once(text,
                "  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();",
                "  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();\n  p.normwp = w.size() > 14 ? (const uint32_t*)w[14].data_ptr() : nullptr;")
    text = once(text, """        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA, positions, kperm3=True) for e in range(lo, lo + cnt)]))
        state["w"].append(torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, positions, kperm3=True) for e in range(lo, lo + cnt)]))""",
                """        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA, positions, kperm3=True) for e in range(lo, lo + cnt)]))
        state["w"].append(torch.cat([_wgs_copy(weights["w2"][e], weights["w2_sf"][e], rowsB, positions, kperm3=True) for e in range(lo, lo + cnt)]))
        nw = weights["post_norm_w"].contiguous().view(-1, 16)
        state["w"].append(torch.stack((nw[:, :8], nw[:, 8:]), dim=-1).contiguous())  # [r16,g,(g,g+8)], exact bf16 bits""")
    text = once(text, """DIAG_VARIANTS = {
    "no-wgmma": tuple(MAIN_DEFINES) + ("-DOKC_WGSX=1",),
    "raw-dequant": tuple(MAIN_DEFINES) + ("-DOKC_WGSRAWDQ=1",),
}""", """DIAG_VARIANTS = {
    "normpair": tuple(MAIN_DEFINES) + ("-DOKC_WGSEPINORM=1",),
}""")
    text = once(text, """            graphs = {name: capture(fn) for name, fn in fns.items()}

            times = {name: [] for name in variants}""", """            graphs = {name: capture(fn) for name, fn in fns.items()}
            torch.cuda.synchronize()
            ref = outs["main"]
            for name in ratios:
                same = torch.equal(ref, outs[name])
                rel = float((outs[name].float() - ref.float()).norm() / ref.float().norm())
                print(f"[diag] parity bs={bs:3d} seed={seed0 + bs} {name}: exact={same} rel={rel:.3e}", flush=True)
                assert same and rel == 0.0

            times = {name: [] for name in variants}""")
    TARGET.write_text(text)
    print(f"wrote {TARGET.name}: {len(text):,} chars, SHA256={hashlib.sha256(TARGET.read_bytes()).hexdigest().upper()}")


if __name__ == "__main__":
    main()
