"""Generate v160: extend the protected v150 RS128 tournament to O m=128."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "submission_rs128_faithful_v150.py"
OUT = ROOT / "submission_rs128_o_v160.py"

base_bytes = BASE.read_bytes()
source = base_bytes.decode()

old_targets = '_V150_TARGETS = ("qkv_gate", "pooler", "unpooler")'
new_targets = '_V150_TARGETS = ("qkv_gate", "o", "pooler", "unpooler")'
old_ks = '_V150_KS = {"qkv_gate": 7, "pooler": 4, "unpooler": 2}'
new_ks = '_V150_KS = {"qkv_gate": 7, "o": 2, "pooler": 4, "unpooler": 2}'
old_unchanged = '        for layer in ("o", "router"):'
new_unchanged = '        for layer in ("router",):'

for old in (old_targets, old_ks, old_unchanged):
    if source.count(old) != 1:
        raise RuntimeError(f"unexpected v150 source match count for {old!r}")

source = source.replace(old_targets, new_targets, 1)
source = source.replace(old_ks, new_ks, 1)
source = source.replace(old_unchanged, new_unchanged, 1)
compile(source, str(OUT), "exec")
OUT.write_bytes(source.encode())

print(OUT)
print("protected_sha256", hashlib.sha256(base_bytes).hexdigest().upper())
print("candidate_sha256", hashlib.sha256(source.encode()).hexdigest().upper())
