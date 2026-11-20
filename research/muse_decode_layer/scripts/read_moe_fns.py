with open("../muse_moe_w4a16/submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if any(k in l for k in ["def generate_input", "def kernel_fn", "def prepare", "def _ma_", "def run"]):
        print(f"Line {i+1}: {l.strip()}")
