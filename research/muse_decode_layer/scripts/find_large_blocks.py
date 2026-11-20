with open('submission.py', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

import re

# find all multiline string assignments
pattern = re.compile(r'([A-Za-z0-9_]+)\s*=\s*(r?["\']{3}[\s\S]*?["\']{3})')
matches = pattern.findall(text)
print(f"Total multiline string variables: {len(matches)}")
for name, val in sorted(matches, key=lambda x: len(x[1]), reverse=True)[:20]:
    print(f"{name:30s}: {len(val):,} bytes ({len(val)/1024:.1f} KB)")
