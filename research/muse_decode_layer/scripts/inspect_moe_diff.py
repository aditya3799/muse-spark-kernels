import os

path_layer = "submission.py"
path_moe = "../muse_moe_w4a16/submission.py"

print("Checking existence of muse_moe_w4a16/submission.py:")
print("Exists:", os.path.exists(path_moe))
if os.path.exists(path_moe):
    size_moe = os.path.getsize(path_moe)
    print(f"Size of moe submission.py: {size_moe} bytes")
    with open(path_moe, "r", encoding="utf-8", errors="ignore") as f:
        moe_lines = f.readlines()
    print(f"Total lines in moe submission.py: {len(moe_lines)}")
    # Print first 50 lines of moe submission.py
    for i in range(min(50, len(moe_lines))):
        if any(k in moe_lines[i] for k in ["MOE_", "OKC_", "VERSION", "def ", "class ", "WGS"]):
            print(f"{i+1}: {moe_lines[i].strip()}")
