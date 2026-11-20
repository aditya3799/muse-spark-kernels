with open('submission_s119_moe_wsi.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract _MA_ALT_SRCS['moe_helper0']
idx1 = text.find('_MA_ALT_SRCS["moe_helper0"] = ')
idx2 = text.find('\nMOE_MID_DEFINES +=', idx1)
code1 = text[idx1:idx2]

# Extract _t2_wsi_patch and its call
idx3 = text.find('def _t2_wsi_patch(c):')
idx4 = text.find('\nMOE_TINY2_DEFINES =', idx3)
code2 = text[idx3:idx4]

locs = {'_MA_ALT_SRCS': {}}
exec(code1, {}, locs)
exec(code2, {}, locs)

print('Patch test successful! _MA_ALT_SRCS["t2_wsi"] length:', len(locs['_MA_ALT_SRCS']['t2_wsi']))
