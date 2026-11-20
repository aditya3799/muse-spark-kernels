with open('../inference_research/keell0renz_muse/kernels/muse_moe_w4a16.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's inspect TINY2_DEFINES in keell0renz muse_moe_w4a16.py
for line in text.split('\n'):
    if 'TINY2_DEFINES =' in line:
        print('keell0renz TINY2_DEFINES:', line)
    if 'PS_SHAPES =' in line:
        print('keell0renz PS_SHAPES:', line)

# And in submission.py:
with open('submission.py', 'r', encoding='utf-8') as f:
    sub_text = f.read()
for line in sub_text.split('\n'):
    if 'MOE_TINY2_DEFINES =' in line:
        print('submission MOE_TINY2_DEFINES:', line)
