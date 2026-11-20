with open('submission.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'def _bfp_gemm', text)]
for idx in matches:
    print(text[idx:idx+1500])
