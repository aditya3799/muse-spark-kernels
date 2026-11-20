"""Mechanically rebase the validated m256 generators on complete v279."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_generator(script_name, base_name, output_name):
    path = ROOT / script_name
    source = path.read_text(encoding="utf-8")
    base_line = next(line for line in source.splitlines()
                     if line.startswith("BASE = ROOT / "))
    output_line = next(line for line in source.splitlines()
                       if line.startswith("OUTPUT = ROOT / "))
    source = source.replace(
        base_line, f'BASE = ROOT / "{base_name}"', 1)
    source = source.replace(
        output_line, f'OUTPUT = ROOT / "{output_name}"', 1)
    namespace = {"__file__": str(path), "__name__": "__main__"}
    exec(compile(source, str(path), "exec"), namespace)


run_generator(
    "generate_pooler256_margin_v228.py",
    "submission_plain_complete_pairplanes_v279.py",
    "submission_plain_v279_pooler256_v280.py",
)
run_generator(
    "generate_o256_config_tournament_v243.py",
    "submission_plain_complete_pairplanes_v279.py",
    "submission_plain_v279_o256_v281.py",
)
run_generator(
    "generate_qkv256_seed_tournament_v255.py",
    "submission_plain_complete_pairplanes_v279.py",
    "submission_plain_v279_qkv256_v282.py",
)
run_generator(
    "generate_o256_config_tournament_v243.py",
    "submission_plain_v279_pooler256_v280.py",
    "submission_plain_v279_pooler256_o256_v283.py",
)
run_generator(
    "generate_qkv256_seed_tournament_v255.py",
    "submission_plain_v279_pooler256_o256_v283.py",
    "submission_plain_v279_full_m256_v284.py",
)
