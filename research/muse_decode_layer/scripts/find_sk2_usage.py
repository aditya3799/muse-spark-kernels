import re

text = open('profiles/submission_s109_cont1.2a4609207ae6410e.txt', encoding='utf-8', errors='ignore').read()

profs = text.split("profile ")
for p in profs[1:]:
    header = p.splitlines()[0]
    sk2_lines = [l for l in p.splitlines() if 'sk2_kernel' in l]
    if sk2_lines:
        print(f"--- profile {header} ---")
        for l in sk2_lines:
            print("  ", l.strip())
