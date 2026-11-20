import glob, json

for f in sorted(glob.glob('check_*.json') + glob.glob('submit_*.json')):
    try:
        with open(f) as fp:
            d = json.load(fp)
        r = d.get('result', d)
        records = [x for x in r.get('records', []) if x.get('event') == 'benchmark']
        v1_s = [x['median_ns']/1000.0 for x in records if x['case']['load'] == 'verify 1' and x['case']['layer'] == 'sliding']
        v1_f = [x['median_ns']/1000.0 for x in records if x['case']['load'] == 'verify 1' and x['case']['layer'] == 'full']
        d1_s = [x['median_ns']/1000.0 for x in records if x['case']['load'] == 'decode 1' and x['case']['layer'] == 'sliding']
        score = r.get("score_us", 0)
        print(f"{f:<25}: score={score:.2f} | d1_sl={d1_s[0] if d1_s else 0:.2f} | v1_sl={v1_s[0] if v1_s else 0:.2f} | v1_fl={v1_f[0] if v1_f else 0:.2f}")
    except Exception as e:
        pass
