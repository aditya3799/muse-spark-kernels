"""Build v172: frozen-v139 WGS plus validated-v167 TINY/TINY2/MID kernels.

The v139 CUDA source, main defines, large-shape weight layout, and M>=17
dispatch remain unchanged.  M<=16 uses a separate v139 _POS-packed state and
the exact small-kernel CUDA sources/defines validated in v167.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_v139_wgs_kperm3.py"
ADVANCED = ROOT / "submission_v167_keell0renz_890809c.py"
DESTINATION = ROOT / "submission_v172_hybrid_small.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def literal_assignments(path: Path) -> dict[str, object]:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return out


def helper0_source(src: str) -> str:
    text = src.replace("threadIdx.x", "TIDX")
    anchor = "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);"
    pos = text.index(anchor)
    pos = text.index("\n", pos) + 1
    block = ("#ifndef OKC_WSH0\n#define OKC_WSH0 0\n#endif\n#if OKC_WSH0\n"
             "#define TIDX ((int)((threadIdx.x + NT) % NTL))\n"
             "#else\n#define TIDX ((int)threadIdx.x)\n#endif\n")
    return text[:pos] + block + text[pos:]


def main() -> None:
    text = BASE.read_text()
    advanced = literal_assignments(ADVANCED)
    var_src = advanced["VAR_SRC"]
    helper_src = helper0_source(var_src)
    patch_kind, edits = advanced["PS_PATCHES"]["t2_wsi"]
    assert patch_kind == "var"
    wsi_src = var_src
    for old, new in edits:
        assert wsi_src.count(old) >= 1, old[:100]
        wsi_src = wsi_src.replace(old, new)

    tiny = tuple(advanced["TINY_DEFINES"]) + ("-DOKC_DF=0", "@v167_var")
    tiny2 = tuple(advanced["TINY2_DEFINES"]) + ("-DOKC_DF=0", "@v167_wsi")
    mid = tuple(advanced["MID_DEFINES"]) + ("-DOKC_WSH0=1", "@v167_helper0")

    text = replace_once(
        text,
        "ALT_SRCS = {}\n",
        "ALT_SRCS = {}\n"
        f"ALT_SRCS['v167_var'] = {var_src!r}\n"
        f"ALT_SRCS['v167_wsi'] = {wsi_src!r}\n"
        f"ALT_SRCS['v167_helper0'] = {helper_src!r}\n",
        "validated small sources",
    )
    text = replace_once(
        text,
        'TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")  # bf16 WSB at NW=8 + WSBL tail (Oct 26: bs1 +2.5% / +3.5%)\n',
        f"TINY_DEFINES = {tiny!r}  # validated v167 tiny source on the _POS state\n",
        "tiny defines",
    )
    text = replace_once(
        text,
        "TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')  # bf16 WSB (Oct 26: bs8 +2.0..2.5%)\n",
        f"TINY2_DEFINES = {tiny2!r}  # validated v167 M8 WSI source on the _POS state\n",
        "tiny2 defines",
    )
    text = replace_once(
        text,
        "MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_SKSC2=1', '-DOKC_FTSC2=1')  # bf16 split-job MID + fast scales (measured +0.8% at bs16)\n",
        f"MID_DEFINES = {mid!r}  # validated v167 helper0/DF MID source on the _POS state\n",
        "mid defines",
    )

    text = replace_once(
        text,
        '    defines = tuple(MAIN_DEFINES if defines is None else defines) + ("-DOKC_MULDEC=" + str(int(use_mul)),)\n'
        '    if DQ7 and "-DOKC_DQ7=1" not in defines:\n',
        '    defines = tuple(MAIN_DEFINES if defines is None else defines)\n'
        '    v167_small = any(d.startswith("@v167_") for d in defines)\n'
        '    if not v167_small:\n'
        '        defines = defines + ("-DOKC_MULDEC=" + str(int(use_mul)),)\n'
        '    if DQ7 and "-DOKC_DQ7=1" not in defines:\n',
        "source-specific define policy",
    )
    text = replace_once(
        text,
        '            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-v,-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",\n'
        '                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={MAIN_NS if major >= 9 else 1}"]),\n'
        '                               f"-DOKC_FNST={4 if major >= 9 else 3}",\n'
        '                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],\n',
        '            extra_cuda_cflags=["-O3",\n'
        '                               *([] if v167_small else ["--use_fast_math", "-Xptxas", "-v,-O3"]),\n'
        '                               f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",\n'
        '                               *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={MAIN_NS if major >= 9 else 1}"]),\n'
        '                               *([] if v167_small else [f"-DOKC_FNST={4 if major >= 9 else 3}"]),\n'
        '                               *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],\n',
        "source-specific compiler policy",
    )

    text = replace_once(
        text,
        '    mods = [tuple(_main_defines(state["sfx"])) + state["sfx"]]\n'
        '    for d in (TINY_DEFINES, TINY2_DEFINES, MID_DEFINES):\n'
        '        if d and (state["sfx"] or "-DOKC_WGS=1" not in tuple(d)):  # the WGS build needs the PRMT scale decode\n'
        '            mods.append(tuple(d) + state["sfx"])\n'
        '    mods = [d for d in mods if (d != tuple(TINY_DEFINES) + state["sfx"])] if use_mul else [tuple(TINY_DEFINES) + state["sfx"]]\n',
        '    if use_mul:\n'
        '        mods = [tuple(_main_defines(state["sfx"])) + state["sfx"]]\n'
        '    else:\n'
        '        mods = [tuple(d) + state["sfx"] for d in (TINY_DEFINES, TINY2_DEFINES, MID_DEFINES) if d]\n',
        "separate layout compile tree",
    )
    text = replace_once(
        text,
        '    state = state["tiny"] if data["hidden_states"].shape[0] <= TINY_MAX_M else state["larger"]\n',
        '    state = state["tiny"] if data["hidden_states"].shape[0] <= MID_MAX_M else state["larger"]\n',
        "small state dispatch",
    )

    DESTINATION.write_text(text)
    print(f"wrote {DESTINATION.name} ({len(text)} characters)")


if __name__ == "__main__":
    main()
