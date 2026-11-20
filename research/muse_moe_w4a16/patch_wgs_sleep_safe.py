with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py', 'rb') as f:
    raw = f.read()

# Let's inspect line endings
has_crlf = b'\r\n' in raw
nl = b'\r\n' if has_crlf else b'\n'
print("Has CRLF:", has_crlf)

code = raw.decode('utf-8')

# 1. Add WGS_SLEEP definition to CUDA_SRC
c_tgt = f"#ifndef OKC_WGSFINSS{code[code.find('#ifndef OKC_WGSFINSS') + 20: code.find('#ifndef OKC_WGSFINSS') + 21]}" # just find substring
idx = code.find('#ifndef OKC_WGSFINSS')
end_idx = code.find('#endif', idx) + 6
c_block = code[idx:end_idx]
print("c_block found:", repr(c_block))

c_def = f"#ifndef OKC_WGS_SLEEP\n#define OKC_WGS_SLEEP 32\n#endif\n#define WGS_SLEEP() do {{ if (OKC_WGS_SLEEP) __nanosleep(OKC_WGS_SLEEP); }} while (0)\n"
if has_crlf:
    c_def = c_def.replace('\n', '\r\n')

code = code[:idx] + c_def + code[idx:]

# 2. Add WGS_SLEEP definition to ALT_SRCS['v167_main'] (escaped newlines)
v_tgt = r"#ifndef OKC_B4FIN\n#define OKC_B4FIN 0\n#endif"
v_rep = r"#ifndef OKC_B4FIN\n#define OKC_B4FIN 0\n#endif\n#ifndef OKC_WGS_SLEEP\n#define OKC_WGS_SLEEP 32\n#endif\n#define WGS_SLEEP() do { if (OKC_WGS_SLEEP) __nanosleep(OKC_WGS_SLEEP); } while (0)"
assert v_tgt in code, "v_tgt not found in v167_main"
code = code.replace(v_tgt, v_rep, 1)

# 3. Replace nanosleep(32)
for t, r in [
    ("if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }",
     "if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) WGS_SLEEP(); }"),
    ("if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }",
     "if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) WGS_SLEEP(); }"),
    ("if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }",
     "if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) WGS_SLEEP(); }"),
    ("if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }",
     "if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) WGS_SLEEP(); }"),
    ("if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }",
     "if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) WGS_SLEEP(); }")
]:
    c = code.count(t)
    code = code.replace(t, r)
    print(f"Replaced {c} occurrences of: {t[:40]}...")

# 4. Add -DOKC_WGS_SLEEP=0 to V167_M32_DEFINES
t_def = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=32', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1')"
r_def = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=32', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1', '-DOKC_WGS_SLEEP=0')"
assert t_def in code, "t_def not found"
code = code.replace(t_def, r_def, 1)

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py', 'w', encoding='utf-8', newline='') as f:
    f.write(code)

print("Patch applied successfully and cleanly!")
