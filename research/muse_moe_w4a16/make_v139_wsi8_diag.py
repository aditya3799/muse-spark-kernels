"""Generate a clean v139 M8 dedicated ring-issuer A/B diagnostic.

The official kernel_fn path remains byte-for-byte equivalent to frozen v139.
Only prepare-time diagnostics compile and time the OKC_WSI=1 candidate.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v164_wgs_polysilu_diag.py"
DST = ROOT / "submission_v139_wsi8_diag.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = SRC.read_text()

    text = replace_once(
        text,
        "#ifndef OKC_WS\n"
        "#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue\n"
        "#endif\n",
        "#ifndef OKC_WS\n"
        "#define OKC_WS 0  // tiny build: one extra helper warp issues the ring's TMA copies and runs every skinny epilogue\n"
        "#endif\n"
        "#ifndef OKC_WSI\n"
        "#define OKC_WSI 0  // diagnostic: dedicated ring-issuer warp in the skinny WS build\n"
        "#endif\n",
        "WSI macro",
    )

    text = replace_once(
        text,
        "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);  // + helper warp (WS) | weights and x producer warps (WGS)  // launched: NT compute threads (+ helper warp | WGS producer warp + epilogue warpgroup)\n",
        "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_WSI ? 64 : 32) : 0);  // WSI adds a dedicated ring-issuer warp\n",
        "launch width",
    )

    text = replace_once(
        text,
        "      }\n"
        "    }\n"
        "#if OKC_NACC == 4 && !OKC_XNOMMA\n",
        "      }\n"
        "    }\n"
        "#if OKC_WSI\n"
        "    __syncwarp();\n"
        "    if (lane == 0) mbar_arrive1(S.wse + sl);  // release slot before accumulator reduction\n"
        "#endif\n"
        "#if OKC_NACC == 4 && !OKC_XNOMMA\n",
        "early slot release",
    )

    text = replace_once(
        text,
        "      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n"
        "      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n",
        "      if (!OKC_WSI && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);\n"
        "      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);   // WSBL: the last units stay with warps 0..K-1\n",
        "disable compute-warp refill",
    )

    text = replace_once(
        text,
        "  if (lane == 0)\n"
        "    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n",
        "  if (!OKC_WSI && lane == 0)\n"
        "    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n",
        "disable helper prefill",
    )

    text = replace_once(
        text,
        "  __threadfence();\n"
        "  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n"
        "}\n"
        "#endif\n\n",
        "  __threadfence();\n"
        "  WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier\n"
        "}\n\n"
        "#if OKC_WSI\n"
        "// Dedicated issuer: prefill the ring, then refill a slot as soon as every compute warp releases it.\n"
        "__device__ void ws_issuer(const Params& p, Smem& S, unsigned char* ring) {\n"
        "  const int lane = threadIdx.x & 31;\n"
        "  WS_JOIN();\n"
        "  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];\n"
        "  const int nA = a1 - a0, N = nA + (b1 - b0);\n"
        "  if (lane == 0) {\n"
        "    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n"
        "    for (int u = OKC_NS; u < N; ++u) {\n"
        "      const int q = u - OKC_NS;\n"
        "      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));\n"
        "      ws_issue(p, S, u, nA, a0, b0, ring);\n"
        "    }\n"
        "  }\n"
        "  __syncwarp();\n"
        "  WS_JOIN();\n"
        "}\n"
        "#endif\n"
        "#endif\n\n",
        "issuer function",
    )

    text = replace_once(
        text,
        "#if OKC_WS\n"
        "  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n"
        "#endif\n",
        "#if OKC_WS\n"
        "#if OKC_WSI\n"
        "  if (tid >= NT + 32) { ws_issuer(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n"
        "#endif\n"
        "  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }\n"
        "#endif\n",
        "issuer dispatch",
    )

    text = replace_once(
        text,
        'DIAG_VARIANTS = {\n    "poly-silu": tuple(MAIN_DEFINES) + ("-DOKC_WGSPOLY=1",),\n}\n',
        'DIAG_VARIANTS = {\n    "wsi": tuple(TINY2_DEFINES) + ("-DOKC_WSI=1",),\n}\n',
        "diagnostic variant",
    )
    text = replace_once(text, "    shapes = (32, 128, 256)\n", "    shapes = (8,)\n", "M8 shape")
    text = replace_once(
        text,
        "    seeds = tuple(1234 + 1009 * i for i in range(7))\n",
        "    seeds = tuple(1234 + 1009 * i for i in range(9))\n",
        "nine seeds",
    )
    text = replace_once(
        text,
        '    variants = {"main": tuple(MAIN_DEFINES) + sfx}\n',
        '    variants = {"main": tuple(TINY2_DEFINES) + sfx}\n',
        "M8 control",
    )
    text = replace_once(text, "            for rep in range(15):\n", "            for rep in range(21):\n", "timing reps")
    text = replace_once(
        text,
        '                if rel > 0.020 or max_row > 0.060:\n'
        '                    raise RuntimeError(f"{name} output drift: rel={rel:.6f}, max_row={max_row:.6f}")\n',
        '                if not torch.equal(outs[name], outs["main"]):\n'
        '                    mismatches = int(torch.count_nonzero(outs[name] != outs["main"]).item())\n'
        '                    raise RuntimeError(f"{name} is not bit-exact: mismatches={mismatches}, rel={rel:.6f}, max_row={max_row:.6f}")\n',
        "bit-exact gate",
    )
    text = replace_once(
        text,
        "        promote = (med_ratio[32] <= 0.995 and med_ratio[128] <= 1.002 and\n"
        "                   med_ratio[256] <= 1.002 and mean_ratio <= 0.9975 and\n"
        "                   max_rel <= 0.020 and max_row <= 0.060)\n",
        "        worst_ratio = max(ratios[name][8])\n"
        "        promote = (med_ratio[8] <= 0.980 and wins[8] >= 7 and worst_ratio <= 1.005 and\n"
        "                   max_rel == 0.0 and max_row == 0.0)\n",
        "M8 promotion gate",
    )
    text = replace_once(
        text,
        '              f"max_row={max_row:.4%}  PROMOTE={promote}", flush=True)\n',
        '              f"max_row={max_row:.4%}  worst={worst_ratio:.4f}x  PROMOTE={promote}", flush=True)\n',
        "report worst pair",
    )

    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
