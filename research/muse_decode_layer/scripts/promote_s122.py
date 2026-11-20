import shutil

shutil.copyfile('submission_s122_att_qdir_pk2.py', 'submission.py')

with open('submission.py', 'rb') as f:
    b = f.read()

print(f"submission.py size: {len(b)} bytes")
print(f"Margin under 1MB: {1000000 - len(b)} bytes")
has_crlf = b'\r\n' in b
has_cr = b'\r' in b
print(f"Has CRLF? {has_crlf}")
print(f"Has lone CR? {has_cr}")
assert len(b) < 1000000, "File too large!"
assert b'\r' not in b, "CRLF found!"
print("Promotion to submission.py verified successfully!")
