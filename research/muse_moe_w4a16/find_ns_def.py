with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("ALT_SRCS['v167_helper0']")
end = text.find("ALT_SRCS['v167_main']")
h0_src = text[pos:end].replace("\\n", "\n")

for i, l in enumerate(h0_src.splitlines()):
    if "OKC_NS" in l and ("define" in l or "constexpr" in l):
        print(f"Line {i}: {l}")
