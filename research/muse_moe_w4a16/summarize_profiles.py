import glob, json, os

files = sorted(glob.glob('profiles/*.json'))
relevant = [f for f in files if any(k in f for k in ['v167', 'v172', 'v177', 'v178', 'v179', 'v182', 'v183', 'v184', 'v185', 'v186', 'v187', 'v188'])]

hdr = f"{'Profile File':<48} | {'Speedup':<7} | {'M=1':<6} | {'M=8':<6} | {'M=16':<6} | {'M=32':<6} | {'M=128':<7} | {'M=256':<7}"
print(hdr)
print('-' * len(hdr))

for f in relevant:
    try:
        d = json.load(open(f))
        sp = d['summary'].get('speedup', 0.0)
        bm = d.get('benchmarks', [])
        by_bs = {b['case']['bs']: b.get('median_ns', 0) / 1000.0 for b in bm if 'case' in b and 'bs' in b['case']}
        m1 = f"{by_bs.get(1, 0):.2f}"
        m8 = f"{by_bs.get(8, 0):.2f}"
        m16 = f"{by_bs.get(16, 0):.2f}"
        m32 = f"{by_bs.get(32, 0):.2f}"
        m128 = f"{by_bs.get(128, 0):.2f}"
        m256 = f"{by_bs.get(256, 0):.2f}"
        fn = os.path.basename(f)
        if len(fn) > 48: fn = fn[:45] + '...'
        print(f"{fn:<48} | {sp:<7.4f} | {m1:<6} | {m8:<6} | {m16:<6} | {m32:<6} | {m128:<7} | {m256:<7}")
    except Exception as e:
        print('ERR', f, e)
