with open('submission.py') as f:
    src = f.read()

idx = src.find("ALT_SRCS['v167_helper0']")
h_src = src[idx:]
start_quote = h_src.find("'")
end_quote = h_src.find("'\ndef ", start_quote + 1)
raw_str = h_src[start_quote+1:end_quote]
decoded = raw_str.encode('utf-8').decode('unicode_escape')

for i, line in enumerate(decoded.split('\n')):
    if 'INTER_SENT' in line or 'inter_sentinel' in line or 'DF' in line:
        if any(w in line for w in ['#define', 'void', 'for', 'if', 'return', '128']):
            print(f"{i}: {line[:120]}")
