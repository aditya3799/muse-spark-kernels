"""Build S020: S010 small-token MoE plus the standalone v99 WGS backend for T > 32.

The winner is never edited.  Large unused v49 CUDA strings are pruned from the
candidate to stay below the challenge's single-file source limit.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import re
import tokenize
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission.py"
MOE_SOURCE = ROOT.parent / "muse_moe_w4a16" / "submission_v99.py"
DESTINATION = ROOT / "submission_s020_v99_wgs.py"
S010_SHA256 = "42bc26b8a02637131513acab6443d2f70a034f0eb681dcc63532bad61a279d69"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def prune_raw_string(text: str, name: str) -> str:
    marker = f'{name} = r"""'
    start = text.index(marker)
    body = start + len(marker)
    end = text.index('"""', body)
    return text[:start] + f'{name} = ""' + text[end + 3 :]


def prefixed_v99() -> str:
    source = MOE_SOURCE.read_text(encoding="utf-8")
    start = source.index("CUDA_SRC = r'''")
    end = source.index("# ---------------------------------------------------------------------------------------------- diagnostics", start)
    source = source[start:end]

    # The standalone file carries a separate bandwidth-diagnostic extension;
    # S020 neither defines nor calls its diagnostic driver.
    bw_start = source.index("BW_SRC = r'''")
    bw_body = bw_start + len("BW_SRC = r'''")
    bw_end = source.index("'''", bw_body)
    source = source[:bw_start] + "BW_SRC = ''\n" + source[bw_end + 3 :]

    # This candidate uses only the main WGS build.  The proven S010 gk backend
    # retains responsibility for T <= 32.
    source = re.sub(r"^TINY_DEFINES = .*$", "TINY_DEFINES = ()", source, flags=re.MULTILINE)
    source = re.sub(r"^TINY2_DEFINES = .*$", "TINY2_DEFINES = ()", source, flags=re.MULTILINE)
    source = re.sub(r"^MID_DEFINES = .*$", "MID_DEFINES = ()", source, flags=re.MULTILINE)

    # Remove comment-only/blank lines, including CUDA // comments, without
    # touching preprocessor directives or executable source.
    source = "\n".join(
        line
        for line in source.splitlines()
        if line.strip()
        and not line.lstrip().startswith("//")
        and not line.lstrip().startswith("# ")
    ) + "\n"

    cuda_marker = "CUDA_SRC = r'''"
    cuda_start = source.index(cuda_marker)
    cuda_body = cuda_start + len(cuda_marker)
    cuda_end = source.index("'''", cuda_body)
    cuda_payload = base64.b64encode(zlib.compress(source[cuda_body:cuda_end].encode("utf-8"), 9)).decode("ascii")
    source = source[:cuda_start] + f'CUDA_SRC = _zsrc("{cuda_payload}")' + source[cuda_end + 3 :]

    tree = ast.parse(source)
    globals_to_prefix: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            globals_to_prefix.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    globals_to_prefix.add(target.id)

    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string in globals_to_prefix:
            token = tokenize.TokenInfo(token.type, "V99_" + token.string, token.start, token.end, token.line)
        tokens.append(token)
    source = tokenize.untokenize(tokens)

    # Python 3.8's tokenizer exposes an f-string as one STRING token, so the
    # namespace pass above cannot see names inside its replacement fields.
    # MAIN_NS is the only embedded v99 global referenced from an f-string.
    source = replace_once(
        source,
        "{MAIN_NS if major >= 9 else 1}",
        "{V99_MAIN_NS if major >= 9 else 1}",
        "v99 f-string namespace",
    )
    source = source.replace(
        'name=f"okc_moe_selmul_',
        'name=f"decode_s020_v99_moe_selmul_',
    )
    return source


winner = SOURCE.read_bytes()
actual = hashlib.sha256(winner).hexdigest()
if actual != S010_SHA256:
    raise RuntimeError(f"submission.py is not protected S010: {actual}")

decoder = winner.decode("utf-8")

# The candidate never calls these old v49/v52 paths.  Keep their Python names
# so dormant helper definitions still parse, but remove hundreds of KB of dead
# CUDA source before adding v99.
for cuda_name in ("M_CUDA", "M_CUDA_L", "M_CUDA_F", "M_CUDA_G", "M_CUDA_X"):
    decoder = prune_raw_string(decoder, cuda_name)

decoder = replace_once(
    decoder,
    "# E027: E025 plus a fixed-4096 pre-expert RMSNorm kernel.",
    "# S020: S010 small-token MoE plus standalone v99 WGS for T > 32.",
    "candidate banner",
)

v99 = prefixed_v99()
decoder = replace_once(
    decoder,
    "\ndef _marlin_bank(weights):",
    "\n\n# S020 embeds the standalone v99 WGS backend under a V99_ prefix.\n"
    + v99
    + "\ndef _marlin_bank(weights):",
    "v99 insertion",
)

# Compile only the four S010 gk variants that serve T <= 32.  All larger
# shapes use v99, so the old ring and wx binaries are dead.
old_compile = '''    with ThreadPoolExecutor(7) as ex:
        list(ex.map(lambda f: f(), [
            lambda: M__extL(0),
            lambda: M__extX(("-DNCW=16",)),
            lambda: M__extX(("-DNCW=8", "-DRPW=2", "-DTPX=32", "-DNSL=2", "-DCPB=2", "-DFGATHER")),
            lambda: M__extK("-DSW=4 -DKS2=1 -DMAXM=2 -DPPS=2"),
            lambda: M__extK("-DSW=4 -DKS2=1 -DMAXM=8"),
            lambda: M__extK("-DSW=2 -DKS2=2"),
            lambda: M__extK("-DSW=2 -DKS1=2 -DKS2=4 -DMAXM=32"),
        ]))
'''
new_compile = '''    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda f: f(), [
            lambda: M__extK("-DSW=4 -DKS2=1 -DMAXM=2 -DPPS=2"),
            lambda: M__extK("-DSW=4 -DKS2=1 -DMAXM=8"),
            lambda: M__extK("-DSW=2 -DKS2=2"),
            lambda: M__extK("-DSW=2 -DKS1=2 -DKS2=4 -DMAXM=32"),
        ]))
'''
decoder = replace_once(decoder, old_compile, new_compile, "small-only v49 compilation")

prepare_anchor = '''    pre_expert = _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS)
    state = {
'''
prepare_addition = '''    pre_expert = _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS)
    v99_layout = V99__prepare_layout({
        "w13": weights["w13"],
        "w13_sf": weights["w13_sf"],
        "w13_gs": weights["w13_gs"],
        "w2": weights["w2"],
        "w2_sf": weights["w2_sf"],
        "w2_gs": weights["w2_gs"],
        "post_norm_w": weights["post_expert_norm"],
    }, True)
    state = {
'''
decoder = replace_once(decoder, prepare_anchor, prepare_addition, "v99 prepare layout")
decoder = replace_once(
    decoder,
    '        "moe_v49": M_prepare({',
    '        "moe_v99_large": {"tiny": v99_layout, "larger": v99_layout},\n        "moe_v49": M_prepare({',
    "v99 prepared state",
)

old_dispatch = '''    expert_output = M_kernel_fn(
        {
            "hidden_states": normalized_expert_input,
            # v49 now localizes global IDs inside its existing route pass;
            # packed weights remain a zero-based bank of this rank's 32 experts.
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "route_start": data["local_start"],
            "config": {"local_start": 0},
        },
        state["moe_v49"],
    )
'''
new_dispatch = '''    moe_data = {
        "hidden_states": normalized_expert_input,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "route_start": data["local_start"],
        "config": {"local_start": 0},
    }
    if tokens > 32:
        # v99's packed bank is local expert 0..31.  This temporary arithmetic
        # is intentionally kept separate for the first performance proof; a
        # winning result will fuse the route offset into v99's route pass.
        expert_output = V99_kernel_fn(
            {
                "hidden_states": normalized_expert_input,
                "topk_ids": (topk_ids - data["local_start"]) % N_EXPERTS,
                "topk_weights": topk_weights,
                "config": {"local_start": 0, "local_count": LOCAL_EXPERTS},
            },
            state["moe_v99_large"],
        )
    else:
        expert_output = M_kernel_fn(moe_data, state["moe_v49"])
'''
decoder = replace_once(decoder, old_dispatch, new_dispatch, "large-M v99 dispatch")

# Keep comfortably below both decimal and binary interpretations of the 1 MB
# source cap.  Preserve the required problem marker; remove only blank and
# comment-only lines (never #include/#define/#pragma directives).
lines = decoder.splitlines()
decoder = lines[0] + "\n" + "\n".join(
    line
    for line in lines[1:]
    if line.strip()
    and not line.lstrip().startswith("//")
    and not line.lstrip().startswith("# ")
) + "\n"

DESTINATION.write_bytes(decoder.encode("utf-8"))
print(f"wrote {DESTINATION} ({len(decoder):,} bytes)")
