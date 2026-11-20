import json

def get_profile(path, prof_idx):
    with open(path) as f:
        text = f.read()
    start = text.find(f"profile {prof_idx}:")
    if start == -1: return ""
    end = text.find(f"profile {prof_idx+1}:", start)
    return text[start:end if end != -1 else start+1500]

print("=== S108 Profile 2 (decode 16 sliding) ===")
print(get_profile('profiles/submission.c3d4cf6955bf4cdc.txt', 2))
print("=== S109 Profile 2 (decode 16 sliding) ===")
print(get_profile('profiles/submission_s109_cont1.2a4609207ae6410e.txt', 2))
