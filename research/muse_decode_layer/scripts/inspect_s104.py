import json

with open('check_s104.json') as f:
    d = json.load(f)

print("Job ID:", d.get('job_id') or d.get('id'))
print("Status:", d.get('status'))
print("Score us:", d.get('score_us'))
print("Speedup:", d.get('speedup'))
print("Summary:", d.get('summary'))
print("Result summary:", d.get('result', {}).get('summary') if 'result' in d else None)
if 'checks' in d:
    print("Checks:", d.get('checks'))
elif 'result' in d and 'checks' in d['result']:
    print("Checks:", d['result']['checks'])
