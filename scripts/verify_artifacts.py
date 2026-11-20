#!/usr/bin/env python3
"""CPU-only verification of exported selected source. Does not submit or run GPU code."""
from pathlib import Path
import ast
import hashlib
import json
root = Path(__file__).resolve().parents[2]
provenance = json.loads((root / "kernel-track/results/source_provenance.json").read_text())
for name, record in provenance.items():
    path = root / "kernel-track" / record["export_file"]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"], str(path)
    ast.parse(data, filename=str(path))
    print(name + ": hash and Python syntax OK")
patch = root / "serving-track/submission/patch.diff"
assert hashlib.sha256(patch.read_bytes()).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
source_hashes = json.loads((root / "serving-track/submission/source_hashes.json").read_text())
for name, expected in source_hashes.items():
    path = root / "serving-track/source" / name
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected, name
    ast.parse(data, filename=name)
print("serving: exact patch and all 10 source hashes/syntax OK")
print("This checks artifact integrity only; GPU correctness/performance is not rerun.")
print("MoE's score-to-source identity remains unverified; see its README.")
