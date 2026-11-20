"""Build an isolated decoder candidate using the validated v49 MoE backend."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission.py"
MOE_SOURCE = ROOT.parent / "muse_moe_w4a16" / "submission_v49.py"
DESTINATION = ROOT.parent / "scratch" / "muse_decode_e024" / "submission.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


decoder = SOURCE.read_text(encoding="utf-8")
moe = MOE_SOURCE.read_text(encoding="utf-8")
moe = moe.split("\n", 1)[1]  # Drop the component problem marker.
# Stay below OKC's 1 MB single-source limit without changing executable code.
# This also removes comment-only lines inside the embedded CUDA raw strings.
moe = "\n".join(
    line
    for line in moe.splitlines()
    if line.strip()
    and not line.lstrip().startswith("//")
    and not line.lstrip().startswith("# ")
)

# The component submission already tuned these exact decoder-relevant buckets on
# the target GPU.  Compile only the six selected implementations and install the
# winning table directly, instead of compiling/timing the full search space in
# every decoder setup.
moe = replace_once(moe, "M_TUNE = True", "M_TUNE = False", "disable MoE autotune")
compile_start = moe.index("    with ThreadPoolExecutor(8) as ex:")
compile_end = moe.index("    torch.cuda.synchronize()", compile_start)
compile_selected = '''    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda f: f(), [
            lambda: M__extL(0),
            lambda: M__extX(("-DNCW=16",)),
            lambda: M__extK("-DSW=4 -DKS2=1 -DMAXM=2 -DPPS=2"),
            lambda: M__extK("-DSW=4 -DKS2=1 -DMAXM=8"),
            lambda: M__extK("-DSW=2 -DKS2=2"),
            lambda: M__extK("-DSW=2 -DKS1=2 -DKS2=4 -DMAXM=32"),
        ]))
'''
moe = moe[:compile_start] + compile_selected + moe[compile_end:]
fixed_tactics = '''    st["tune"] = {
        1: ("gk", 0, 0, "-DSW=4 -DKS2=1 -DMAXM=2 -DPPS=2"),
        8: ("gk", 0, 0, "-DSW=4 -DKS2=1 -DMAXM=8"),
        16: ("gk", 0, 0, "-DSW=2 -DKS2=2"),
        32: ("gk", 0, 0, "-DSW=2 -DKS1=2 -DKS2=4 -DMAXM=32"),
        128: ("ring", 1, 0),
        256: ("wx", 0, 0, "-DNCW=16"),
    }
'''
moe = replace_once(moe, "    if M_DIAG:\n", fixed_tactics + "    if M_DIAG:\n", "fixed MoE tactics")

decoder = replace_once(
    decoder,
    "# v20q experiment: warp-sort fused FA3 scheduler metadata for full verification.",
    "# E024: S006 plus the validated fused v49 packed-NVFP4 MoE backend.",
    "candidate banner",
)
decoder = replace_once(
    decoder,
    "\ndef _marlin_bank(weights):",
    "\n\n# E024 embeds the independently validated BF16/NVFP4 v49 expert backend.\n"
    + moe
    + "\n\ndef _marlin_bank(weights):",
    "MoE source insertion",
)
decoder = replace_once(
    decoder,
    '        "bank": _marlin_bank(weights),\n        "humming": _humming_bank(weights),',
    '''        "moe_v49": M_prepare({
            "w13": weights["w13"],
            "w13_sf": weights["w13_sf"],
            "w13_gs": weights["w13_gs"],
            "w2": weights["w2"],
            "w2_sf": weights["w2_sf"],
            "w2_gs": weights["w2_gs"],
            "post_norm_w": weights["post_expert_norm"],
        }),''',
    "MoE prepared state",
)
decoder = replace_once(
    decoder,
    "    small_aligned = None\n    humming_prepared = None\n",
    "",
    "obsolete alignment locals",
)

route_start = decoder.index("        # Route preprocessing belongs to the routing dependency chain.")
route_end = decoder.index("    # This branch depends only on the pooler.", route_start)
decoder = decoder[:route_start] + decoder[route_end:]

record_start = decoder.index("        if small_aligned is not None:")
record_end = decoder.index("    # all-reduce: identity", record_start)
decoder = decoder[:record_start] + decoder[record_end:]

dispatch_start = decoder.index("    if humming_prepared is not None:")
dispatch_end = decoder.index("    if tokens <= SLICED_UNPOOL_MAX_TOKENS:", dispatch_start)
dispatch = '''    expert_output = M_kernel_fn(
        {
            "hidden_states": normalized_expert_input,
            # Decoder weights contain this EP rank's 32 experts only.  Rotate
            # global routing IDs into that local bank before entering v49,
            # whose packed kernels otherwise index weights with local_start.
            "topk_ids": (topk_ids - data["local_start"]) % N_EXPERTS,
            "topk_weights": topk_weights,
            "config": {"local_start": 0},
        },
        state["moe_v49"],
    )
'''
decoder = decoder[:dispatch_start] + dispatch + decoder[dispatch_end:]

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
# Emit repository-native LF bytes even when this builder runs under Windows.
DESTINATION.write_bytes(decoder.encode("utf-8"))
print(f"wrote {DESTINATION} ({len(decoder):,} bytes)")
