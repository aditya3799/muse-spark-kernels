import json
import re

with open('profiles/submission.6cafa69edc014362.json', encoding='utf-8') as f:
    d = json.load(f)

for i, p in enumerate(d.get('profiles', [])):
    case = p.get('case', {})
    name = f"{case.get('load')} {case.get('layer')}"
    report = p.get('report', '')
    gemms = []
    for line in report.splitlines():
        m = re.search(r'^(.*?)\s+([\d\.]+)\s*\S+s\s+([\d\.]+)%', line.strip())
        if m:
            kname = m.group(1).strip()
            tus = float(m.group(2))
            if 'sk2' in kname or 'nvjet' in kname or 'rq<' in kname:
                gemms.append((kname[:50], tus))
    print(f"=== Case {i:2d}: {name:<20} ===")
    for kname, tus in gemms:
        print(f"  {kname:<50} | {tus:5.1f} us")
