"""Build E025 by fusing EP expert-ID localization into the selected v49 kernels."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission.py"
DESTINATION = ROOT.parent / "scratch" / "muse_decode_e025" / "submission.py"


def replace_exact(text: str, old: str, new: str, label: str, expected: int = 1) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{label}: expected {expected} matches, found {count}")
    return text.replace(old, new)


def transform_region(
    text: str,
    start_marker: str,
    end_marker: str,
    transforms: list[tuple[str, str, str, int]],
) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    region = text[start:end]
    for old, new, label, expected in transforms:
        region = replace_exact(region, old, new, label, expected)
    return text[:start] + region + text[end:]


source = SOURCE.read_text(encoding="utf-8")
source = replace_exact(
    source,
    "# E024: S006 plus the validated fused v49 packed-NVFP4 MoE backend.",
    "# E025: E024 plus device-side EP route localization inside v49.",
    "candidate banner",
)

# T <= 32: gk.  The int64 ABI slot now carries the address of the decoder's
# CUDA int32 local_start tensor.  Packed weights and global scales remain local
# arrays, so only route classification reads this offset.
source = transform_region(
    source,
    'M_CUDA_K = r"""',
    "\nM__modK = {}",
    [
        (
            "int M, lo, smem; unsigned long long* prof;",
            "int M, smem; const int* route_start; unsigned long long* prof;",
            "gk parameter struct",
            1,
        ),
        ("p.ids[tid] - p.lo", "p.ids[tid] - p.route_start[0]", "gk route localization", 1),
        (
            "p->lo + sm->act[tt >> 9]",
            "sm->act[tt >> 9]",
            "gk local packed w1",
            1,
        ),
        ("p->lo + sm->act[ca]", "sm->act[ca]", "gk local packed w2", 1),
        ("p.g1[p.lo + e]", "p.g1[e]", "gk local g1", 1),
        ("p.g2[p.lo + e]", "p.g2[e]", "gk local g2", 2),
        (
            "torch::Tensor scr, torch::Tensor out, int64_t lo, torch::Tensor prof)",
            "torch::Tensor scr, torch::Tensor out, int64_t route_start_ptr, torch::Tensor prof)",
            "gk wrapper ABI",
            1,
        ),
        (
            "(int)x.size(0), (int)lo, g_smemK,",
            "(int)x.size(0), g_smemK, (const int*)route_start_ptr,",
            "gk parameter initializer",
            1,
        ),
    ],
)

# T = 128: ring.  Remove every use of the route base from weight/global-scale
# addresses and use it only when converting global expert IDs to [0, 31].
source = transform_region(
    source,
    'M_CUDA_L = r"""',
    "\nM__mod = None",
    [
        (
            "int M, lo, ks; unsigned long long* prof;",
            "int M, ks; const int* route_start; unsigned long long* prof;",
            "ring parameter struct",
            1,
        ),
        ("p.ids[i] - p.lo", "p.ids[i] - p.route_start[0]", "ring route localization i", 1),
        ("p.ids[r] - p.lo", "p.ids[r] - p.route_start[0]", "ring route localization r", 1),
        ("p.g1[p.lo + e]", "p.g1[e]", "ring local g1", 2),
        ("p.g2[p.lo + e]", "p.g2[e]", "ring local g2", 2),
        ("(p.lo + e)", "e", "ring local packed weights", 2),
        (
            "torch::Tensor ws, torch::Tensor out, int64_t lo, int64_t ks, torch::Tensor prof)",
            "torch::Tensor ws, torch::Tensor out, int64_t route_start_ptr, int64_t ks, torch::Tensor prof)",
            "ring wrapper ABI",
            1,
        ),
        (
            "M, (int)lo, (int)ks,",
            "M, (int)ks, (const int*)route_start_ptr,",
            "ring parameter initializer",
            1,
        ),
    ],
)

# T = 256: wx.  Its gather and fused MoE kernel both classify routes, so both
# read the same device-side route base while all packed arrays remain local.
source = transform_region(
    source,
    'M_CUDA_X = r"""',
    "\nM__modX = {}",
    [
        (
            "int M, lo; unsigned long long* prof;",
            "int M; const int* route_start; unsigned long long* prof;",
            "wx parameter struct",
            1,
        ),
        (
            "xgather(const __nv_bfloat16* x, const int* ids, int M, int lo, uint8_t* xg)",
            "xgather(const __nv_bfloat16* x, const int* ids, int M, const int* route_start, uint8_t* xg)",
            "wx gather ABI",
            1,
        ),
        (
            "const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = M * 8;",
            "const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, R = M * 8, lo = route_start[0];",
            "wx gather route load",
            1,
        ),
        ("p.ids[i] - p.lo", "p.ids[i] - p.route_start[0]", "wx route localization i", 1),
        ("p.ids[r] - p.lo", "p.ids[r] - p.route_start[0]", "wx route localization r", 1),
        ("p.g1[p.lo + e]", "p.g1[e]", "wx local g1", 1),
        ("p.g2[p.lo + e]", "p.g2[e]", "wx local g2", 1),
        (
            "torch::Tensor ctr, torch::Tensor out, int64_t lo, torch::Tensor prof)",
            "torch::Tensor ctr, torch::Tensor out, int64_t route_start_ptr, torch::Tensor prof)",
            "wx wrapper ABI",
            1,
        ),
        (
            "ids.data_ptr<int>(), M, (int)lo, xg.data_ptr<uint8_t>()",
            "ids.data_ptr<int>(), M, (const int*)route_start_ptr, xg.data_ptr<uint8_t>()",
            "wx gather launch",
            1,
        ),
        (
            "ctr.data_ptr<int>(), M, (int)lo,",
            "ctr.data_ptr<int>(), M, (const int*)route_start_ptr,",
            "wx parameter initializer",
            1,
        ),
    ],
)

# Use distinct extension names so local and judge-side JIT caches cannot reuse
# an E024 binary with the old ABI.
source = replace_exact(source, '"moe_w4a16_v48K"', '"moe_w4a16_e025K"', "gk module name")
source = replace_exact(source, '"moe_w4a16_v30L"', '"moe_w4a16_e025L"', "ring module name")
source = replace_exact(source, '"moe_w4a16_v48X"', '"moe_w4a16_e025X"', "wx module name")

source = replace_exact(
    source,
    'out, data["config"]["local_start"],\n                                              state["prof"][:0] if prof is None else prof)',
    'out, data["route_start"].data_ptr(),\n                                              state["prof"][:0] if prof is None else prof)',
    "gk Python launch",
)
source = replace_exact(
    source,
    'out, data["config"]["local_start"], state.get("profW", state["prof"]))',
    'out, data["route_start"].data_ptr(), state.get("profW", state["prof"]))',
    "wx Python launch",
)
source = replace_exact(
    source,
    'state["wsL"], out, data["config"]["local_start"], ks if M <= M_KSL_MAX else 1, state["prof"])',
    'state["wsL"], out, data["route_start"].data_ptr(), ks if M <= M_KSL_MAX else 1, state["prof"])',
    "ring Python launch",
)

source = replace_exact(
    source,
    '''            # Decoder weights contain this EP rank's 32 experts only.  Rotate
            # global routing IDs into that local bank before entering v49,
            # whose packed kernels otherwise index weights with local_start.
            "topk_ids": (topk_ids - data["local_start"]) % N_EXPERTS,
            "topk_weights": topk_weights,
            "config": {"local_start": 0},''',
    '''            # v49 now localizes global IDs inside its existing route pass;
            # packed weights remain a zero-based bank of this rank's 32 experts.
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "route_start": data["local_start"],
            "config": {"local_start": 0},''',
    "decoder dispatch",
)

# No selected CUDA source may retain the conflated route/weight offset field.
for start_marker, end_marker, label in (
    ('M_CUDA_K = r"""', "\nM__modK = {}", "gk"),
    ('M_CUDA_L = r"""', "\nM__mod = None", "ring"),
    ('M_CUDA_X = r"""', "\nM__modX = {}", "wx"),
):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    if "p.lo" in source[start:end] or "p->lo" in source[start:end]:
        raise RuntimeError(f"{label}: stale route/weight offset use remains")

DESTINATION.parent.mkdir(parents=True, exist_ok=True)
DESTINATION.write_bytes(source.encode("utf-8"))
print(f"wrote {DESTINATION} ({len(source):,} bytes)")
