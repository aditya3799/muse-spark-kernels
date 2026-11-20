with open('submission_s081_keell0renz_890809c.py') as f:
    s081 = f.read()

with open('submission_s103_rq_cgemv.py') as f:
    s103 = f.read()

import re
print("S081 has B4FIN:", "B4FIN" in s081)
print("S103 has B4FIN:", "B4FIN" in s103)
print("Count of B4FIN in s081:", s081.count("B4FIN"))
print("Count of B4FIN in s103:", s103.count("B4FIN"))
