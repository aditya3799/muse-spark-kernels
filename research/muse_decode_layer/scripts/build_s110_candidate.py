with open('submission_s109_cont1.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update module name
text = text.replace('muse_att_v298w1pf_s109_{arch}', 'muse_att_v298w1pf_s110_{arch}')

# 2. In first _moe(M)
old_b4_1 = '''    if (M is not None and 16 < M <= 32 and "_main_defines" in globals()
            and tuple(_main_defines(_MA_SFX)) == tuple(MOE_DEFINES)):  # bet4_claude: decode 32 gets the B4FIN WGS build
        if _MOE_B4 is None:
            _MOE_B4 = _moe_build(tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",))
        return _MOE_B4'''
assert old_b4_1 in text, "old_b4_1 not found"
text = text.replace(old_b4_1, '    # S110: M=32 falls through to main WGS (0 local memory spill)')

# 3. In second _moe(M, state)
old_b4_2 = '''    elif M is not None and 16 < M <= 32 and tuple(_main_defines(sfx)) == tuple(MOE_DEFINES):
        key, defines = "b4", tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)'''
assert old_b4_2 in text, "old_b4_2 not found"
text = text.replace(old_b4_2, '    # S110: M=32 falls through to main')

# 4. In _mf_cases
old_mf = '''def _mf_cases(sfx):
    main = tuple(_main_defines(sfx))
    return [(32, tuple(MOE_DEFINES) + ("-DOKC_B4FIN=1",)), (128, main), (256, main)]'''
assert old_mf in text, "old_mf not found"
new_mf = '''def _mf_cases(sfx):
    main = tuple(_main_defines(sfx))
    return [(32, main), (128, main), (256, main)]'''
text = text.replace(old_mf, new_mf)

out_path = 'submission_s110_cont1_nob4fin.py'
with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

import os
size = os.path.getsize(out_path)
print(f"Successfully wrote {out_path}, size = {size} bytes")
assert size < 1000000, f"Size {size} exceeds 1,000,000 bytes!"
