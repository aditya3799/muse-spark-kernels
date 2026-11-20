import re
import ast

with open("submission.py") as f:
    orig = f.read()

# Check that submission.py is currently valid python syntax
ast.parse(orig)
print("Original submission.py is valid Python syntax.")

# Check the targets to replace in ALT_SRCS['v167_main']
m = re.search(r"ALT_SRCS\['v167_main'\]\s*=\s*'((?:[^'\\]|\\.)*)'", orig, re.DOTALL)
assert m, "ALT_SRCS['v167_main'] not found!"

src = m.group(1).encode('utf-8').decode('unicode_escape')

target1 = "*reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);\\n      CTA_SYNC();"
print("Target 1 found in raw orig?", target1 in orig)

target_m32 = "V167_M32_DEFINES = ('-DOKC_NW=16', '-DOKC_FAST1=0', '-DOKC_EPI4=0', '-DOKC_WGS=1', '-DOKC_FT_T=32', '-DOKC_MMAX=64', '-DOKC_NO_BIG=1', '-DOKC_WGSNS=4', '-DOKC_RLOAD=0', '-DOKC_WPBURST=0', '-DOKC_B4FIN=1')"
print("Target M32 found in orig?", target_m32 in orig)
