import json

with open('check_s104.json') as f:
    d = json.load(f)

res = d.get('result', {})
print("Log tail:")
print(res.get('log_tail'))
