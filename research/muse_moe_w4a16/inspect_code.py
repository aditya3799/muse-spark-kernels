with open("submission_v169_wgs_grid_tune.py") as f:
    s = f.read()

import re
print("grid in v169:")
for l in s.splitlines():
    if "grid" in l.lower():
        print(l[:100])
