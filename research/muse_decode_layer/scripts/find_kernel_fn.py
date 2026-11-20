with open('submission_s109_cont1.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'def kernel_fn', text)]
for idx in matches:
    print('--- AT INDEX', idx, '---')
    print(text[idx:idx+2500])
