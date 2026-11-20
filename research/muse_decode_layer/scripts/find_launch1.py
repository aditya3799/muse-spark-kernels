with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'launch<1>', text)]
for idx in matches:
    print(text[idx-100:idx+400])
    print("="*40)
