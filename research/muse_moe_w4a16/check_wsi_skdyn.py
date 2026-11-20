with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("ALT_SRCS['v167_wsi']")
end = text.find("ALT_SRCS['v167_helper0']")
wsi_src = text[pos:end].replace("\\n", "\n")

for i, l in enumerate(wsi_src.splitlines()):
    if "SKDYN" in l or "OKC_TDYN" in l:
        print(f"Line {i}: {l}")
