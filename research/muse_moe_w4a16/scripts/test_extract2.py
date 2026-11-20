with open("submission.py") as f:
    orig = f.read()

idx = orig.find("ALT_SRCS['v167_main'] = '")
v_idx = orig.find("V167_MAIN_DEFINES", idx)
print(f"Start: {idx}, V167_MAIN_DEFINES at: {v_idx}")
print("Between end of string and V167_MAIN_DEFINES:")
print(repr(orig[v_idx-30:v_idx+30]))
