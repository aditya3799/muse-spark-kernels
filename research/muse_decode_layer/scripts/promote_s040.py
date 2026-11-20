"""Promote the exact checked S040 bytes to submission.py under hash guards."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "submission.py"
CANDIDATE = ROOT / "submission_s040_vjeux_f35a41ee.py"
EXPECTED_ACTIVE = "4b40bd21366077da9a3d969b5ca876fc67db308767fad60e7bc535cab111b2b6"
EXPECTED_CANDIDATE = "e4ccd1407b704fc9605dd882750c313bf6c68287d8e5da14fbeba2300074bd43"


active_bytes = ACTIVE.read_bytes()
candidate_bytes = CANDIDATE.read_bytes()
active_hash = hashlib.sha256(active_bytes).hexdigest()
candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
if active_hash != EXPECTED_ACTIVE:
    raise RuntimeError(f"refusing to overwrite unexpected active source: {active_hash}")
if candidate_hash != EXPECTED_CANDIDATE:
    raise RuntimeError(f"refusing to promote unexpected S040 source: {candidate_hash}")
ACTIVE.write_bytes(candidate_bytes)
promoted_hash = hashlib.sha256(ACTIVE.read_bytes()).hexdigest()
if promoted_hash != EXPECTED_CANDIDATE:
    raise RuntimeError(f"promotion verification failed: {promoted_hash}")
print(f"promoted S040 to {ACTIVE}")
print("sha256", promoted_hash)
