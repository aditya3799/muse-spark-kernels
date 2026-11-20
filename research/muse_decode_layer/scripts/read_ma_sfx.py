with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'ma_sfx', text)]
for idx in matches:
    print(text[idx-50:idx+150])
    print("-" * 30)
