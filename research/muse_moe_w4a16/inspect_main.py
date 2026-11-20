with open('submission.py') as f:
    src = f.read()

idx = src.find("ALT_SRCS['v167_main']")
h_src = src[idx:]
start_quote = h_src.find("'")
end_quote = h_src.find("'\n", start_quote + 1)
raw_str = h_src[start_quote+1:end_quote]
decoded = raw_str.encode('utf-8').decode('unicode_escape')

lines = decoded.split('\n')
for i in range(2665, 2700):
    print(f"{i}: {lines[i]}")
