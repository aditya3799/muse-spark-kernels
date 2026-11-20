with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re
m = re.search(r'def _prepare_layout.*?(?=def _prepare_v167)', text, re.DOTALL)
if m:
    for line in m.group(0).splitlines():
        if 'grid' in line:
            print(line)
