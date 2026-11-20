with open('submission_s111_cont_l1_only.py') as f:
    text = f.read()

idx = text.find('if constexpr (Cfg<L, FLAT>::WS && CONT)')
print(text[max(0, idx-500):min(len(text), idx+500)])
