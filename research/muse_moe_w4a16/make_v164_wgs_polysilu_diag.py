"""Generate the v164 WGS-only polynomial-SiLU diagnostic from v150."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "submission_v150_wgs_compute_diag.py"
DST = ROOT / "submission_v164_wgs_polysilu_diag.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = SRC.read_text()

    text = replace_once(
        text,
        "#ifndef OKC_WGSRAWDQ\n#define OKC_WGSRAWDQ 0  // dev: feed the packed code word directly to WGMMA; diagnostic garbage output\n#endif\n",
        "#ifndef OKC_WGSRAWDQ\n#define OKC_WGSRAWDQ 0  // dev: feed the packed code word directly to WGMMA; diagnostic garbage output\n#endif\n"
        "#ifndef OKC_WGSPOLY\n#define OKC_WGSPOLY 0  // dev: polynomial SiLU only in the WGS phase-A epilogue\n#endif\n",
        "WGS polynomial macro",
    )
    text = replace_once(
        text,
        "#define SILU_DIV(a, b) ((a) / (b))\n#endif\n",
        "#define SILU_DIV(a, b) ((a) / (b))\n#endif\n"
        "__device__ __forceinline__ float wgs_silu(float x) {\n"
        "#if OKC_WGSPOLY\n"
        "  const float z = x * x;\n"
        "  const float p = fmaf(fmaf(2.82057459e-4f, z, -1.16906086e-2f), z, 2.34261343e-1f);\n"
        "  return x * __saturatef(fmaf(x, p, .5f));\n"
        "#else\n"
        "  return SILU_DIV(x, 1.f + __expf(-x));\n"
        "#endif\n"
        "}\n",
        "WGS polynomial helper",
    )
    text = replace_once(
        text,
        "        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;\n",
        "        const float v = wgs_silu(gt) * up;\n",
        "WGS phase-A SiLU call",
    )
    text = replace_once(
        text,
        'DIAG_VARIANTS = {\n    "no-wgmma": tuple(MAIN_DEFINES) + ("-DOKC_WGSX=1",),\n    "raw-dequant": tuple(MAIN_DEFINES) + ("-DOKC_WGSRAWDQ=1",),\n}\n',
        'DIAG_VARIANTS = {\n    "poly-silu": tuple(MAIN_DEFINES) + ("-DOKC_WGSPOLY=1",),\n}\n',
        "diagnostic variants",
    )
    text = replace_once(
        text,
        '    medians = {name: {bs: [] for bs in shapes} for name in variants}\n',
        '    medians = {name: {bs: [] for bs in shapes} for name in variants}\n'
        '    errors = {name: {bs: [] for bs in shapes} for name in variants if name != "main"}\n',
        "diagnostic error storage",
    )
    text = replace_once(
        text,
        '            graphs = {name: capture(fn) for name, fn in fns.items()}\n\n            times = {name: [] for name in variants}\n',
        '            graphs = {name: capture(fn) for name, fn in fns.items()}\n'
        '            ref = outs["main"].float()\n'
        '            ref_norm = torch.linalg.vector_norm(ref).clamp_min(1e-12)\n'
        '            ref_rows = torch.linalg.vector_norm(ref, dim=1).clamp_min(1e-12)\n'
        '            for name in ratios:\n'
        '                delta = outs[name].float() - ref\n'
        '                rel = float((torch.linalg.vector_norm(delta) / ref_norm).item())\n'
        '                max_row = float((torch.linalg.vector_norm(delta, dim=1) / ref_rows).max().item())\n'
        '                errors[name][bs].append((rel, max_row))\n'
        '                if rel > 0.020 or max_row > 0.060:\n'
        '                    raise RuntimeError(f"{name} output drift: rel={rel:.6f}, max_row={max_row:.6f}")\n'
        '\n'
        '            times = {name: [] for name in variants}\n',
        "diagnostic output comparison",
    )
    text = replace_once(
        text,
        '            detail = "  ".join(f"{name}={med[name]:.1f}us ({med[name] / med[\'main\']:.4f}x)" for name in ratios)\n'
        '            print(f"[diag] bs={bs:3d} seed={seed0 + bs} touched={touched:2d} main={med[\'main\']:.1f}us  {detail}", flush=True)\n',
        '            detail = "  ".join(\n'
        '                f"{name}={med[name]:.1f}us ({med[name] / med[\'main\']:.4f}x; "\n'
        '                f"rel={errors[name][bs][-1][0]:.4%}, row={errors[name][bs][-1][1]:.4%})"\n'
        '                for name in ratios\n'
        '            )\n'
        '            print(f"[diag] bs={bs:3d} seed={seed0 + bs} touched={touched:2d} main={med[\'main\']:.1f}us  {detail}", flush=True)\n',
        "diagnostic detail",
    )
    text = replace_once(
        text,
        '        mean_ratio = sum(all_ratios) / len(all_ratios)\n'
        '        print(f"[diag] summary {name}: {\'  \'.join(fields)}  ratios {\'  \'.join(rfields)}  mean={mean_ratio:.4f}x", flush=True)\n',
        '        mean_ratio = sum(all_ratios) / len(all_ratios)\n'
        '        max_rel = max(rel for bs in shapes for rel, _ in errors[name][bs])\n'
        '        max_row = max(row for bs in shapes for _, row in errors[name][bs])\n'
        '        wins = {bs: sum(r < 1.0 for r in ratios[name][bs]) for bs in shapes}\n'
        '        med_ratio = {bs: sorted(ratios[name][bs])[len(ratios[name][bs]) // 2] for bs in shapes}\n'
        '        promote = (med_ratio[32] <= 0.995 and med_ratio[128] <= 1.002 and\n'
        '                   med_ratio[256] <= 1.002 and mean_ratio <= 0.9975 and\n'
        '                   max_rel <= 0.020 and max_row <= 0.060)\n'
        '        print(f"[diag] summary {name}: {\'  \'.join(fields)}  ratios {\'  \'.join(rfields)}  "\n'
        '              f"mean={mean_ratio:.4f}x  wins={wins}  max_rel={max_rel:.4%} "\n'
        '              f"max_row={max_row:.4%}  PROMOTE={promote}", flush=True)\n',
        "diagnostic promotion gate",
    )

    DST.write_text(text)
    print(f"wrote {DST.name} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
