with open('submission.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'OKC_CONT', text)]
for idx in matches:
    print('--- AT INDEX', idx, '---')
    print(text[max(0, idx-60):min(len(text), idx+200)])
