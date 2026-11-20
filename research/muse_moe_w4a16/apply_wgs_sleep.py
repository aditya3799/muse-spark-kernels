with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py') as f:
    code = f.read()

# 1. Add #ifndef OKC_WGS_SLEEP in v167_main
target1 = r"#ifndef B4_EARLY_Y\n#define B4_EARLY_Y 1\n#endif"
repl1 = r"#ifndef B4_EARLY_Y\n#define B4_EARLY_Y 1\n#endif\n#ifndef OKC_WGS_SLEEP\n#define OKC_WGS_SLEEP 32\n#endif\n#define WGS_SLEEP() do { if (OKC_WGS_SLEEP) __nanosleep(OKC_WGS_SLEEP); } while (0)"

assert target1 in code, "target1 not found"
code = code.replace(target1, repl1, 1)

# 2. Replace the 3 nanosleep(32) in v167_main
t1 = r"if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }"
r1 = r"if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) WGS_SLEEP(); }"
assert t1 in code, "t1 not found"
code = code.replace(t1, r1, 1)

t2 = r"if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }"
r2 = r"if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) WGS_SLEEP(); }"
assert t2 in code, "t2 not found"
code = code.replace(t2, r2, 1)

t3 = r"if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }"
r3 = r"if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) WGS_SLEEP(); }"
assert t3 in code, "t3 not found"
code = code.replace(t3, r3, 1)

# 3. Add -DOKC_WGS_SLEEP=0 to V167_M32_DEFINES
t_def = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=32', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1')"
r_def = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=32', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1', '-DOKC_WGS_SLEEP=0')"
assert t_def in code, "t_def not found"
code = code.replace(t_def, r_def, 1)

with open('/mnt/d/OpenKernelChallenge/muse_moe_w4a16/submission.py', 'w') as f:
    f.write(code)

print("Successfully applied OKC_WGS_SLEEP=0 patch to submission.py!")
