with open('submission_s081_keell0renz_890809c.py', 'r', encoding='utf-8') as f:
    s081 = f.read()

with open('scripts/build_s103_candidate.py', 'r', encoding='utf-8') as f:
    s103_build = f.read()

# Check size of S081
print("S081 size:", len(s081.encode('utf-8')))

# Check size of exact-C12 GEMV code from submission_s090_s081_cgemv_m1.py
with open('submission_s090_s081_cgemv_m1.py', 'r', encoding='utf-8') as f:
    s090 = f.read()

marker = "# S073: protected bf16_proj v105 exact-C12 GEMV for T=1 QKV and sliced unpool."
appendix_pos = s090.rfind("# ---------------------------------------------------------------------------", 0, s090.index(marker))
cgemv_code = s090[appendix_pos:]
print("CGEMV code size:", len(cgemv_code.encode('utf-8')))

total_size_estimate = len(s081.encode('utf-8')) + len(cgemv_code.encode('utf-8')) + 1000
print("Total size estimate with S081 + CGEMV:", total_size_estimate)
print("Fits under 1,000,000?", total_size_estimate < 1000000)
print("Remaining headroom:", 1000000 - total_size_estimate)
