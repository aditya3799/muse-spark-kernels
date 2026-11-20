with open('submission_s120_wsi_fpart.py', 'rb') as f:
    b = f.read()

assert b'\r' not in b, "CR found in source!"
assert len(b) < 1000000, f"File too large: {len(b)}"

with open('submission.py', 'wb') as f:
    f.write(b)

print(f"Copied to submission.py: {len(b)} bytes, no CR.")
