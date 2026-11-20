"""Rebase the complete v249/v250 cache mechanisms on validated v279."""

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
    "generate_cgemv_cache_margin_v249.py",
    "submission_plain_complete_pairplanes_v279.py",
    "submission_plain_v279_cgemv_cache_v287.py",
)
run_generator(
    "generate_pooler1_cache_tournament_v250.py",
    "submission_plain_v279_cgemv_cache_v287.py",
    "submission_plain_v279_cgemv_poolcache_v288.py",
)
