with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'def _moe\(', text)]
for idx in matches:
    print(text[idx:idx+1200])
    print("="*50)
