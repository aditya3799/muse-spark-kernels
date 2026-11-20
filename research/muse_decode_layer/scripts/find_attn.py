with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'attn_kernel', text)]
print(f"Total attn_kernel occurrences: {len(matches)}")
for idx in matches[:5]:
    start = max(0, idx - 100)
    end = min(len(text), idx + 200)
    print("="*40)
    print(text[start:end])
