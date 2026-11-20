with open('submission_s113_rq_qkv_fixed.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'RQPROD', text)]
for idx in matches:
    print('--- AT INDEX', idx, '---')
    print(text[max(0, idx-50):min(len(text), idx+200)])
