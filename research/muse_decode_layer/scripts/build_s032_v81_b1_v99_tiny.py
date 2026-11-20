"""Build S032: S031 plus standalone-v99's measured T=1/T=8 paths."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_s031_v81_b1.py"
DESTINATION = ROOT / "submission_s032_v81_b1_v99_tiny.py"
SOURCE_SHA256 = "f5518e5bb7b78321da571d7479aa931db654af1f468ac01d6196cb6d581530ae"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


source_bytes = SOURCE.read_bytes()
actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
if actual_sha256 != SOURCE_SHA256:
    raise RuntimeError(
        f"unexpected S031 source hash: {actual_sha256} != {SOURCE_SHA256}"
    )
candidate = source_bytes.decode("utf-8")

candidate = replace_once(
    candidate,
    "V99_TINY_DEFINES = ()",
    'V99_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", '
    '"-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", '
    '"-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", '
    '"-DOKC_FINPF=1")',
    "v99 T1 specialization",
)
candidate = replace_once(
    candidate,
    "V99_TINY2_DEFINES = ()",
    'V99_TINY2_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", '
    '"-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4", "-DOKC_UWARP=0", '
    '"-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", '
    '"-DOKC_FINPF=1")',
    "v99 T8 specialization",
)

old_prepare = '''    v99_layout = V99__prepare_layout({
        "w13": weights["w13"],
        "w13_sf": weights["w13_sf"],
        "w13_gs": weights["w13_gs"],
        "w2": weights["w2"],
        "w2_sf": weights["w2_sf"],
        "w2_gs": weights["w2_gs"],
        "post_norm_w": weights["post_expert_norm"],
    }, True)
'''
new_prepare = '''    v99_weights = {
        "w13": weights["w13"],
        "w13_sf": weights["w13_sf"],
        "w13_gs": weights["w13_gs"],
        "w2": weights["w2"],
        "w2_sf": weights["w2_sf"],
        "w2_gs": weights["w2_gs"],
        "post_norm_w": weights["post_expert_norm"],
    }
    v99_tiny_layout = V99__prepare_layout(v99_weights, False)
    v99_layout = V99__prepare_layout(v99_weights, True)
'''
candidate = replace_once(
    candidate, old_prepare, new_prepare, "separate tiny v99 weight layout"
)
candidate = replace_once(
    candidate,
    '"moe_v99_large": {"tiny": v99_layout, "larger": v99_layout},',
    '"moe_v99_large": {"tiny": v99_tiny_layout, "larger": v99_layout},',
    "tiny layout state",
)
candidate = replace_once(
    candidate,
    "    if tokens > 32:\n",
    "    if tokens <= 8 or tokens > 32:\n",
    "tiny v99 dispatch",
)

ast.parse(candidate)
encoded = candidate.encode("utf-8")
if len(encoded) >= 1_000_000:
    raise RuntimeError("S032 exceeds the challenge source limit")
DESTINATION.write_bytes(encoded)
print(f"wrote {DESTINATION} ({len(encoded):,} bytes)")
print("sha256", hashlib.sha256(encoded).hexdigest())
