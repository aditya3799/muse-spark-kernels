with open('submission_v172_hybrid_small.py') as f1, open('submission.py') as f2:
    s1, s2 = f1.read(), f2.read()

# search for TINY_DEFINES in both
for name, s in [('v172', s1), ('v178', s2)]:
    for line in s.split('\n'):
        if 'TINY_DEFINES' in line:
            print(name, line)
        if 'TINY_MAX_M' in line:
            print(name, line)
