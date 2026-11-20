with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

def get_block(name):
    start_tag = f"ALT_SRCS['{name}'] = '"
    start = text.find(start_tag)
    if start == -1:
        return ""
    start += len(start_tag)
    next_alt = text.find("\nALT_SRCS['", start)
    if next_alt != -1:
        end = text.rfind("'\n", start, next_alt)
    else:
        next_def = text.find("\ndef ", start)
        end = text.rfind("'\n", start, next_def)
    raw = text[start:end]
    return raw.replace("\\n", "\n")

wsi = get_block("v167_wsi")
h0 = get_block("v167_helper0")
main = get_block("v167_main")

lines_wsi = wsi.splitlines()
lines_h0 = h0.splitlines()
lines_main = main.splitlines()

print(f"wsi: {len(lines_wsi)} lines")
print(f"h0: {len(lines_h0)} lines")
print(f"main: {len(lines_main)} lines")

print("\n--- Header / Defines in wsi (first 40 lines) ---")
for l in lines_wsi[:40]:
    if l.startswith("#define") or l.startswith("#ifndef"):
        print("wsi:", l)

print("\n--- Header / Defines in h0 (first 40 lines) ---")
for l in lines_h0[:40]:
    if l.startswith("#define") or l.startswith("#ifndef"):
        print("h0: ", l)

print("\n--- Header / Defines in main (first 40 lines) ---")
for l in lines_main[:40]:
    if l.startswith("#define") or l.startswith("#ifndef"):
        print("main: ", l)
