with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

print("v167_helper0 present?", "ALT_SRCS['v167_helper0']" in text)
