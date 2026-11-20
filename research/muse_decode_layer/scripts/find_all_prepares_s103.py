with open('submission_s103_rq_cgemv.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'def prepare\(', text)]
for idx in matches:
    print("="*40)
    print(text[idx:idx+400])
