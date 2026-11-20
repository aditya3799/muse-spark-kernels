with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'\.run\(', text)]
print("Total .run( calls:", len(matches))
for pos in matches[:10]:
    line_start = text.rfind('\n', 0, pos) + 1
    line_end = text.find('\n', pos)
    print(text[line_start:line_end].strip()[:100])
