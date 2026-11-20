with open('submission_s111_cont_l1_only.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'bool CONT', text)]
for idx in matches:
    print('--- AT INDEX', idx, '---')
    print(text[max(0, idx-100):min(len(text), idx+300)])
