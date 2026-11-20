with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

import re

# Replace MID_DEFINES with v190's proven best MID_DEFINES
old_mid = re.search(r'MID_DEFINES = [^\n]+', text).group(0)
new_mid = "MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_WSH0=1', '@v167_helper0')  # v190 proven best 89.04us M=16"

text = text.replace(old_mid, new_mid, 1)

with open('submission.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Successfully set MID_DEFINES to v190 helper0 in submission.py!")
