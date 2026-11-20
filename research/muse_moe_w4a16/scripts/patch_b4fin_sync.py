with open("submission.py") as f:
    orig = f.read()

target = "*reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\\n      CTA_SYNC();\\n    }"
replacement = "*reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\\n      if (it + gridDim.x < p.M * NSEG) CTA_SYNC();\\n    }"

assert target in orig, "Target not found in submission.py"
patched = orig.replace(target, replacement, 1)

with open("submission.py", "w") as f:
    f.write(patched)

print("Successfully patched B4FIN trailing barrier in submission.py!")
