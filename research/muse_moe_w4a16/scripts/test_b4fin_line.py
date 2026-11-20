with open("submission.py") as f:
    orig = f.read()

idx = orig.find("ALT_SRCS['v167_main'] = '")
v_idx = orig.find("V167_MAIN_DEFINES", idx)
v167_main = orig[idx:v_idx]

target = "*reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\\n      CTA_SYNC();"
print("In v167_main?", target in v167_main)
