with open("submission.py", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "attn_kernel<" in l and "true" in l:
        print(f"Line {i+1}: {l.strip()}")
        for j in range(i+1, min(i+15, len(lines))):
            print(f"  {lines[j].strip()}")
            if ";" in lines[j]:
                break
