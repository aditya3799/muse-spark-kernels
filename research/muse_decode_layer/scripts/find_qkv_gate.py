with open('submission_s109_cont1.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'qkv_gate', text)]
print(f"Total qkv_gate matches: {len(matches)}")
for idx in matches[:10]:
    line_start = text.rfind('\n', 0, idx) + 1
    line_end = text.find('\n', idx)
    print(text[line_start:line_end].strip()[:100])
