with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

idx1 = text.find('ATT7_CUDA_SRC =')
idx2 = text.find('ATT7_CPP_SRC =', idx1)
att7_cuda = text[idx1:idx2]

with open('../muse_attention_decode/submission.py', 'r', encoding='utf-8') as f:
    dec_text = f.read()

idx3 = dec_text.find('CUDA_SRC =')
idx4 = dec_text.find('CPP_SRC =', idx3)
dec_cuda = dec_text[idx3:idx4]

print(f"ATT7_CUDA_SRC len: {len(att7_cuda)}")
print(f"muse_attention_decode CUDA_SRC len: {len(dec_cuda)}")
