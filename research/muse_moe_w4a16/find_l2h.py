with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

def get_block(name):
    start_tag = f"ALT_SRCS['{name}'] = '"
    start = text.find(start_tag) + len(start_tag)
    next_alt = text.find("\nALT_SRCS['", start)
    end = text.rfind("'\n", start, next_alt) if next_alt != -1 else text.rfind("'\n", start, text.find("\ndef ", start))
    return text[start:end].replace("\\n", "\n")

main = get_block("v167_main")
for i, l in enumerate(main.splitlines()):
    if "L2H" in l:
        print(f"Line {i}: {l}")
