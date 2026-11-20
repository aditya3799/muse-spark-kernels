with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'SK_CEPI', text)]
for idx in matches:
    print(text[idx-20:idx+80])
