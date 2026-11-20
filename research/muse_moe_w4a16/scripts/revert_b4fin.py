with open("submission.py") as f:
    text = f.read()

target = "*reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\\n      if (it + gridDim.x < p.M * NSEG) CTA_SYNC();\\n    }"
clean = "*reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\\n      CTA_SYNC();\\n    }"

assert target in text, "Target not found"
text = text.replace(target, clean, 1)

with open("submission.py", "w") as f:
    f.write(text)

print("Reverted B4FIN trailing barrier to clean CTA_SYNC().")
