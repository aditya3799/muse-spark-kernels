with open('submission_s111_cont_l1_only.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'\bCONT\b', text)]
print(f"Total CONT occurrences: {len(matches)}")
for idx in matches:
    line_start = text.rfind('\n', 0, idx) + 1
    line_end = text.find('\n', idx)
    print(f"Line: {text[line_start:line_end].strip()}")
