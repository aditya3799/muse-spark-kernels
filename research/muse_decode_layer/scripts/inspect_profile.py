import json

with open('profiles/submission.725c19cbeb9b4d48.json', 'r') as f:
    p = json.load(f)

for prof in p.get('profiles', [])[:6]:
    case = prof['case']
    print("="*60)
    print(f"Index {prof['index']}: Load={case.get('load')}, Layer={case.get('layer')}, Batch={case.get('batch')}, Weight={case.get('weight')}")
    print(prof['report'])
