with open('submission.py') as f:
    text = f.read()

idx = text.find('template <int L, bool F> struct Cfg')
if idx == -1:
    idx = text.find('struct Cfg')
print(text[idx:idx+1500])
