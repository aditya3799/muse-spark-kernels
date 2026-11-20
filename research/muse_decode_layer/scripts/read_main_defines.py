with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'MOE_DEFINES\s*=', text)]
for idx in matches:
    print(text[idx:idx+300])

m2 = re.search(r'def _main_defines\(.*?\n\n', text, re.DOTALL)
if m2:
    print(m2.group(0))
