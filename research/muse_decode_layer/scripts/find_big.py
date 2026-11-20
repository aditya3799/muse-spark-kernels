with open('submission_s081_keell0renz_890809c.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'\bBIG\b', text)]
print(f"Total BIG occurrences: {len(matches)}")
for idx in matches[:15]:
    start = max(0, idx - 50)
    end = min(len(text), idx + 100)
    print(text[start:end])
    print("-" * 30)
