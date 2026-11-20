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
    return text[start:end].replace("\\n", "\n")

main = get_block("v167_main")
lines = main.splitlines()

# Search for B4FIN occurrences in v167_main
print("--- B4FIN occurrences in v167_main ---")
for i, l in enumerate(lines):
    if "B4FIN" in l:
        for j in range(max(0, i - 2), min(len(lines), i + 10)):
            print(f"{j}: {lines[j]}")
        print("...")
