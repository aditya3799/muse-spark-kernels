with open('submission.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r't32s6hzw', text)]
for idx in matches:
    print('--- AT INDEX', idx, '---')
    print(text[max(0, idx-50):min(len(text), idx+150)])
