import json

for fname in ['profiles/submission_s108_att_cgemv.d29c356373ca40a0.json', 'check_s120.json', 'check_s121.json', 'submit_s121.json', 'check_s122.json']:
    try:
        with open(fname) as f:
            d = json.load(f)
        recs = [r for r in (d.get('records') or d.get('result', {}).get('records', [])) if r.get('event') == 'benchmark']
        c6 = recs[6]
        c = c6['case']
        name = f"{c['load']} {c['layer']}"
        print(f"{fname:<55}: {name} = {c6['median_ns']/1000.0:.2f} us")
    except Exception as e:
        print(f"{fname:<55}: {e}")
