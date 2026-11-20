import json

for s in ['102', '103', '104', '105', '107', '108', '109', '110']:
    try:
        d = json.load(open(f'check_s{s}.json'))
        res = d.get('result', {})
        print(f"S{s}: passed={res.get('passed')}, speedup={res.get('speedup'):.4f}, score={res.get('score_us'):.2f} us")
    except Exception as e:
        print(f"S{s}: {e}")
