with open('submission_v187_combined_b4_large.py') as f:
    src = f.read()

idx1 = src.find("ALT_SRCS['v167_main']")
idx2 = src.find("ALT_SRCS['v182_decoder_b4']")

s1 = src[idx1:idx2]
s2 = src[idx2:src.find("def _stored_pos", idx2)]

s1_code = s1[s1.find("'")+1:s1.rfind("'")].encode('utf-8').decode('unicode_escape')
s2_code = s2[s2.find("'")+1:s2.rfind("'")].encode('utf-8').decode('unicode_escape')

import difflib
diff = list(difflib.unified_diff(s1_code.split('\n'), s2_code.split('\n'), fromfile='v167_main', tofile='v182_b4'))
for line in diff[60:150]:
    if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
        print(line[:120])
