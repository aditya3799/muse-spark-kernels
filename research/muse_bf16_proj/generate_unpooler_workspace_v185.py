from pathlib import Path


BASE = Path("submission_plain_exact_v180.py")
SOURCE = Path("submission_lt_workspace_v110.py")
OUT = Path("submission_plain_unpooler_workspace_v185.py")


base = BASE.read_text(encoding="utf-8")
source = SOURCE.read_text(encoding="utf-8")
marker = "# v110: isolated nonzero-workspace cuBLASLt algorithm sweep"
if source.count(marker) != 1:
    raise RuntimeError("unexpected v110 block marker count")
block = source[source.index(marker):]

# Reuse the already validated plain implementation, but isolate the one large
# exact shape that v110 did not cover.  These substitutions affect source text
# at generation time only; the emitted submission contains ordinary Python.
block = block.replace("v110", "v185").replace("V110", "V185")
old_targets = '''_V185_TARGETS = {
    'qkv_gate': (2304, 8192, False),
    'pooler': (4096, 8192, True),
}'''
new_targets = '''_V185_TARGETS = {
    # v110 covered qkv256 and pooler256; unpooler256 was not measured.
    # Use the existing K-major copy selected by the incumbent Lt route.
    'unpooler': (8192, 4096, True),
}'''
if block.count(old_targets) != 1:
    raise RuntimeError("unexpected workspace target anchor count")
block = block.replace(old_targets, new_targets)
block = block.replace("_V185_ROUTE_CORE_RATIO = 0.98",
                      "_V185_ROUTE_CORE_RATIO = 0.97")
block = block.replace("_V185_SCORE_GATE = 0.02",
                      "_V185_SCORE_GATE = 0.0")
block = block.replace(
    "# m=256 projections with the largest remaining arithmetic/traffic headroom.",
    "# previously untested unpooler m=256 projection.")
block = block.replace(
    "# validation, paired cold-L2 timing, and a projected 2% whole-score gate pass.",
    "# validation and a replay-adjusted 3% route-core gate pass.")

text = base.rstrip() + "\n\n\n" + block.lstrip()
OUT.write_text(text, encoding="utf-8", newline="\n")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
