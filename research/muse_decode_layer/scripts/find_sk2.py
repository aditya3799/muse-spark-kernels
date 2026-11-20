with open('submission.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'sk2', text)]
print("Total sk2 occurrences:", len(matches))
for idx in matches[:5]:
    line_start = text.rfind('\n', 0, idx) + 1
    line_end = text.find('\n', idx)
    print(text[line_start:line_end].strip()[:100])
