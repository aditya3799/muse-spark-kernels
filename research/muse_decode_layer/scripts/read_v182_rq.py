text = open(r'd:\OpenKernelChallenge\muse_bf16_proj\submission_plain_composite_v182.py').read()
idx = text.find('rq_go<32, 6')
print(text[idx-400:idx+800])
