with open('submission_s109_cont1.py') as f:
    text = f.read()

import re
matches = re.findall(r'name=f?["\']([a-zA-Z0-9_{}]+)["\']', text)
for name in sorted(set(matches)):
    print(name)
