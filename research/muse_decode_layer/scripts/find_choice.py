with open('submission.py') as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'state\[["\']choice["\']\]', text)]
for idx in matches:
    print(text[max(0, idx-50):min(len(text), idx+300)])
