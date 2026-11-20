with open("submission.py", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("ALT_SRCS['v167_helper0']")
end = text.find("ALT_SRCS['v167_main']")
h0_src = text[pos:end].replace("\\n", "\n")

pos_ws = h0_src.find("skinny_ring_ws")
lines = h0_src[pos_ws:pos_ws + 10000].splitlines()

for i, l in enumerate(lines[:80]):
    if "NW" in l:
        print(f"Line {i}: {l}")
