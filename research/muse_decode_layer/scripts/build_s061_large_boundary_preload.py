import ast
import hashlib
from pathlib import Path


ROOT = Path(r"D:\OpenKernelChallenge\muse_decode_layer")
SOURCE = ROOT / "submission.py"
OUTPUT = ROOT / "submission_s061_large_boundary_preload.py"
EXPECTED = "63af668fbb4c49033be0db7113472518e639ee428c4b9d151cf02a67da39c7c2"

old = '''    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                  num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
'''

new = '''    if T in (128, 256) and width == HIDDEN:
        # S061: these verify shapes have a long O/unpool producer window. Load the
        # immutable norm/gate vectors before the PDL wait, as the proven T=32 path
        # does, so only the branch-dependent reduction remains on the critical tail.
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                      g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                      num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
    if T > 32 and width == HIDDEN:
        _graft_al(br, stream, s_out, h_out)
        _graft_row_boundary[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, HIDDEN, width, post, 4, 16 if T > 192 else 32,
                                  num_warps=16 if T > 192 else 32, launch_pdl=True)
        return True
'''

data = SOURCE.read_bytes()
actual = hashlib.sha256(data).hexdigest()
if actual != EXPECTED:
    raise SystemExit(f"refusing to build from {actual}; expected exact S052 {EXPECTED}")

text = data.decode("utf-8")
if text.count(old) != 1:
    raise SystemExit(f"expected one target block, found {text.count(old)}")

candidate = text.replace(old, new, 1)
ast.parse(candidate)
OUTPUT.write_text(candidate, encoding="utf-8", newline="\n")
print(OUTPUT)
print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
