with open('submission_s111_cont_l1_only.py') as f:
    text = f.read()

idx = text.find('if constexpr (Cfg<L, FLAT>::WS && CONT)')
start = text.rfind('__global__', 0, idx)
print(text[start:start+500])
