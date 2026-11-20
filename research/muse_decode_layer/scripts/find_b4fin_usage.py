with open('submission.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'B4FIN', text)]
print("Total B4FIN occurrences:", len(matches))
for pos in matches:
    line_start = text.rfind('\n', 0, pos) + 1
    line_end = text.find('\n', pos)
    print(text[line_start:line_end].strip()[:100])
