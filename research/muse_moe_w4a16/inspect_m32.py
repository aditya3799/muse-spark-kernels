with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
m = re.search(r'def _prepare_v167_m32_layout.*?(?=def )', text, re.DOTALL)
if m:
    print(m.group(0))
else:
    print("Not found")
