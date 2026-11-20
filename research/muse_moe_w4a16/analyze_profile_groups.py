"""Summarize official shape medians for the frozen-v139 and advanced lineages."""

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "profiles"
SHAPES = (1, 16, 32, 8, 128, 256)
GROUPS = {
    "v139-equivalent": (
        "submission_v139_wgs_kperm3.*.json",
        "submission_v166_wgs_epiaddr_diag.*.json",
        "submission_v168_wgs_qvpos_diag.*.json",
    ),
    "advanced-v167": (
        "submission_v167_keell0renz_890809c.*.json",
        "submission_v168_code_hist_diag.*.json",
        "submission_v167_recheck.*.json",
    ),
    "hybrid-v172": (
        "submission_v172_hybrid_small.*.json",
        "submission_v173_hybrid_promote.*.json",
        "submission_v174_hybrid_retry.*.json",
    ),
    "hybrid-v176-m32": (
        "submission_v176_hybrid_m32_scratch.*.json",
        "submission_v177_hybrid_m32_promote.*.json",
        "submission_v178_hybrid_m32_retry.*.json",
    ),
}


def main() -> None:
    for group, patterns in GROUPS.items():
        paths = []
        for pattern in patterns:
            paths.extend(PROFILES.glob(pattern))
        rows = []
        print(f"\n{group}: {len(paths)} runs")
        for path in sorted(paths):
            data = json.loads(path.read_text())
            values = {row["case"]["bs"]: row["median_ns"] / 1000.0 for row in data["benchmarks"]}
            score = data["summary"]["score_us"]
            rows.append((score, values))
            print(f"  {data['job_id']} score={score:.3f} " +
                  " ".join(f"M{shape}={values[shape]:.2f}" for shape in SHAPES))
        for label, fn in (("mean", statistics.mean), ("median", statistics.median)):
            print(f"  {label:6s} score={fn(row[0] for row in rows):.3f} " +
                  " ".join(f"M{shape}={fn(row[1][shape] for row in rows):.2f}" for shape in SHAPES))


if __name__ == "__main__":
    main()
