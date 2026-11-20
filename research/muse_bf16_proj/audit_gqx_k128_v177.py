"""Static reproducibility audit for the revised v177 K128 GQX candidate."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_clean_composite_v174.py"
GENERATOR = ROOT / "generate_gqx_k128_v177.py"
CANDIDATE = ROOT / "submission_gqx_k128_v177.py"

EXPECTED = {
    BASE: "cc313802f74a9d40666845d5906132996518a37008dbc05ea8b86f519064371e",
    GENERATOR: "ab0fbb57a0488d8a43fd294b630bf3053b3ba37b0b73f6b9e77c3874f8ca8100",
    CANDIDATE: "aaa46a08611f70c2e8a574b136b27b2bdfed397a03927a3fea05c17438affd3a",
}

EXPECTED_CONFIGS = {
    ("qkv_gate", 128): (1, 3, 4),
    ("qkv_gate", 256): (1, 3, 2),
    ("pooler", 128): (1, 2, 4),
    ("pooler", 256): (1, 2, 2),
    ("unpooler", 128): (1, 1, 4),
    ("unpooler", 256): (1, 1, 2),
    ("o", 128): (1, 1, 4),
    ("o", 256): (1, 1, 2),
}

H200_MAX_DYNAMIC_SMEM = 232_448


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"missing assignment: {name}")


def main() -> None:
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{path.name} hash changed: {actual} != {expected}")

    candidate_bytes = CANDIDATE.read_bytes()
    if len(candidate_bytes) >= 1_000_000:
        raise RuntimeError(f"candidate exceeds upload limit: {len(candidate_bytes)} bytes")
    text = candidate_bytes.decode()
    if not text.splitlines() or text.splitlines()[0] != "# okc: problem=muse_bf16_proj":
        raise RuntimeError("problem marker changed")

    tree = ast.parse(text, filename=str(CANDIDATE))
    source = assignment(tree, "GQX_SRC")
    configs = assignment(tree, "GQK_CONFIGS")
    if configs != EXPECTED_CONFIGS:
        raise RuntimeError(f"unexpected GQK configs: {configs!r}")

    # Each K128 helper must expose all eight K16 HGMMAs through one inline-PTX
    # accumulator list. This is the revision that removed PTXAS C7515.
    for bn in (128, 256):
        opcode = f"wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16"
        if source.count(opcode) != 8:
            raise RuntimeError(f"expected eight {opcode} instructions, found {source.count(opcode)}")
    if source.count("wgmma_ss_k128<BN>(acc") != 1:
        raise RuntimeError("K128 mainloop must have one combined inline-PTX call")

    smem_rows = []
    for (layer, bn), (nwg, cluster, stages) in sorted(configs.items()):
        bm = nwg * 64
        stage_bytes = 2 * (bm * 128 + bn * 128)
        fragment_bytes = (bn // 8) * nwg * 128 * 16 if cluster > 1 else 0
        smem = 1024 + stages * stage_bytes + 16 * stages + 16 + fragment_bytes
        if smem > H200_MAX_DYNAMIC_SMEM:
            raise RuntimeError(f"{layer}/BN{bn}/CL{cluster} needs {smem} shared bytes")
        smem_rows.append((layer, bn, cluster, stages, smem))

    print(f"v177 audit OK: {len(candidate_bytes)} bytes")
    print(f"candidate sha256 {EXPECTED[CANDIDATE]}")
    for layer, bn, cluster, stages, smem in smem_rows:
        print(
            f"{layer:10s} BN={bn:3d} CL={cluster} stages={stages} "
            f"smem={smem:6d} headroom={H200_MAX_DYNAMIC_SMEM - smem:5d}"
        )


if __name__ == "__main__":
    main()
