import os

with open("submission_s108_att_cgemv.py", "r", encoding="utf-8") as f:
    src = f.read()

# 1. Update ATT7 build
old_att = 'name=f"muse_att_v298w1pf_s108_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,\n                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=0", "-DOKC_PK2=0", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],'
new_att = 'name=f"muse_att_v298w1pf_s126_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,\n                            functions=["run"], extra_cuda_cflags=["-DOKC_QDIRECT=1", "-DOKC_PK2=1", "-DOKC_W0ENABLE=1", "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],'
assert old_att in src, "Failed to find old_att"
src = src.replace(old_att, new_att, 1)

# 2. Update w0on in attn_kernel
old_w0on = 'const bool w0on = OKC_W0ENABLE && p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;'
new_w0on = 'const bool w0on = OKC_W0ENABLE && L == 8 && p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;'
assert old_w0on in src, "Failed to find old_w0on"
src = src.replace(old_w0on, new_w0on, 1)

# 3. Update fpart in attn_kernel
old_fpart = 'constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 && OKC_WSPLIT == 500 && !OKC_BALPART &&'
new_fpart = 'constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0L1 && OKC_WSPLIT == 500 && !OKC_BALPART &&'
assert old_fpart in src, "Failed to find old_fpart"
src = src.replace(old_fpart, new_fpart, 1)

# 4. Update _plan7
old_plan7 = 'ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else ((3 * sms) if L == 1 else target)'
new_plan7 = 'ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target'
assert old_plan7 in src, "Failed to find old_plan7"
src = src.replace(old_plan7, new_plan7, 1)

# 5. Disable MFAB_LAYER
old_mfab = """_mfin_prior_prepare = prepare


def prepare(weights):
    t0 = _rq_time.time()
    th = _mf_start_base_builds(weights) if MFAB_LAYER else None
    st = _mfin_prior_prepare(weights)
    print(f"[mfab] layer prepare done at {_rq_time.time() - t0:.0f}s", flush=True)
    if MFAB_LAYER and th is not None:
        th.join(timeout=max(1.0, 195.0 - (_rq_time.time() - t0)))
        if _rq_time.time() - t0 < 205.0 and not th.is_alive():
            try:
                _mf_ab(st, t0, weights)
            except Exception as ex:  # diagnostics must never break the run
                print(f"[mfab] A/B failed: {str(ex)[-400:]}", flush=True)
        else:
            print(f"[mfab] skipped at {_rq_time.time() - t0:.0f}s", flush=True)
        torch.cuda.synchronize()
    return st"""

new_mfab = """_mfin_prior_prepare = prepare
prepare = _mfin_prior_prepare"""

assert old_mfab in src, "Failed to find old_mfab"
src = src.replace(old_mfab, new_mfab, 1)

# Ensure strictly Unix LF
src = src.replace("\r\n", "\n").replace("\r", "\n")

out_path = "submission_s126.py"
with open(out_path, "wb") as f:
    f.write(src.encode("ascii"))

sz = os.path.getsize(out_path)
print(f"Built {out_path}: {sz} bytes")
assert sz < 1000000, f"File too large: {sz} bytes >= 1,000,000"
assert b"\r\n" not in open(out_path, "rb").read(), "Found CRLF in output file!"
print("All assertions passed!")
