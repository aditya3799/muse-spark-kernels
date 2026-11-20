import os
import shutil

src = "submission_s123.py"
dst = "submission.py"

assert os.path.exists(src), f"{src} not found"
size = os.path.getsize(src)
print(f"Source size: {size} bytes")
assert size < 1000000, f"File too large: {size} bytes"

with open(src, "rb") as f:
    content = f.read()

assert b"\r\n" not in content, "Found CRLF line endings!"
assert len(content) == size

shutil.copyfile(src, dst)
dst_size = os.path.getsize(dst)
print(f"Copied to {dst}: {dst_size} bytes")
assert dst_size == size

with open(dst, "rb") as f:
    dst_content = f.read()
assert dst_content == content
print("Promotion successful! submission.py is byte-for-byte identical to submission_s123.py.")
