with open('submission.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Revert TINY2_DEFINES
old_tiny2 = "TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=3', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_DF=0', '@v167_wsi')"
new_tiny2 = "TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1', '-DOKC_DF=0', '@v167_wsi')"

if old_tiny2 in text:
    text = text.replace(old_tiny2, new_tiny2)

# Revert MID_DEFINES
old_mid = "MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_DF=0', '@v167_wsi')"
new_mid = "MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1', '-DOKC_DF=0', '@v167_wsi')"

if old_mid in text:
    text = text.replace(old_mid, new_mid)

with open('submission.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Reverted TINY2_DEFINES and MID_DEFINES in submission.py!")
