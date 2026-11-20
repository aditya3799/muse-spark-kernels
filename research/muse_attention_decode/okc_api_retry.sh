#!/usr/bin/env bash
set -euo pipefail

base="https://openkernelchallenge.dev/api/v1"
token="$(/home/aditya/.okc/bin/okc api token)"

request() {
  curl --fail-with-body --silent --show-error \
    --http1.1 \
    --retry 4 --retry-all-errors --retry-delay 2 \
    --connect-timeout 15 --max-time 90 \
    -H "Authorization: Bearer ${token}" "$@"
}

case "${1:-}" in
  check)
    file="${2:?usage: okc_api_retry.sh check FILE}"
    payload="$(mktemp)"
    trap 'rm -f "$payload"' EXIT
    python3 -c 'import json, pathlib, sys; print(json.dumps({"code": pathlib.Path(sys.argv[1]).read_text(), "mode": "check"}))' "$file" > "$payload"
    request -H "Content-Type: application/json" --data-binary "@${payload}" \
      "${base}/problems/muse_attention_decode/submissions"
    ;;
  submit)
    file="${2:?usage: okc_api_retry.sh submit FILE}"
    payload="$(mktemp)"
    trap 'rm -f "$payload"' EXIT
    python3 -c 'import json, pathlib, sys; print(json.dumps({"code": pathlib.Path(sys.argv[1]).read_text()}))' "$file" > "$payload"
    request -H "Content-Type: application/json" --data-binary "@${payload}" \
      "${base}/problems/muse_attention_decode/submissions"
    ;;
  jobs) request "${base}/jobs" ;;
  findhash)
    request "${base}/jobs" |
      python3 -c 'import json, sys
h=sys.argv[1].lower(); j=json.load(sys.stdin); xs=j if isinstance(j,list) else j.get("jobs",[])
for x in xs:
 s=" ".join(str(x.get(k,"")) for k in ("content_hash","payload_ref","code_sha256")).lower()
 if h in s: print(json.dumps({k:x.get(k) for k in ("id","mode","status","created_at","content_hash","payload_ref","result")}))' "${2:?usage: okc_api_retry.sh findhash HASH}"
    ;;
  near)
    request "${base}/jobs" |
      python3 -c 'import json, sys
t=float(sys.argv[1]); j=json.load(sys.stdin); xs=j if isinstance(j,list) else j.get("jobs",[])
for x in sorted(xs,key=lambda y:abs(float(y.get("created_at",0))-t))[:int(sys.argv[2])]:
 r=x.get("result") or {}; print(f"{x.get(chr(105)+chr(100))} {x.get(chr(109)+chr(111)+chr(100)+chr(101))} {x.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))} created={x.get(chr(99)+chr(114)+chr(101)+chr(97)+chr(116)+chr(101)+chr(100)+chr(95)+chr(97)+chr(116))} score={r.get(chr(115)+chr(99)+chr(111)+chr(114)+chr(101)+chr(95)+chr(117)+chr(115))} hash={x.get(chr(99)+chr(111)+chr(110)+chr(116)+chr(101)+chr(110)+chr(116)+chr(95)+chr(104)+chr(97)+chr(115)+chr(104))}")' "${2:?usage: okc_api_retry.sh near CREATED_AT [COUNT]}" "${3:-12}"
    ;;
  status) request "${base}/jobs/${2:?usage: okc_api_retry.sh status JOB_ID}" ;;
  batch)
    shift
    for jid in "$@"; do
      request "${base}/jobs/${jid}" |
        python3 -c 'import json,sys
j=json.load(sys.stdin); r=j.get("result") or {}
print(f"{sys.argv[1]} status={j.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))} mode={j.get(chr(109)+chr(111)+chr(100)+chr(101))} duplicate_of={j.get(chr(100)+chr(117)+chr(112)+chr(108)+chr(105)+chr(99)+chr(97)+chr(116)+chr(101)+chr(95)+chr(111)+chr(102))} passed={r.get(chr(112)+chr(97)+chr(115)+chr(115)+chr(101)+chr(100))} score={r.get(chr(115)+chr(99)+chr(111)+chr(114)+chr(101)+chr(95)+chr(117)+chr(115))} speedup={r.get(chr(115)+chr(112)+chr(101)+chr(101)+chr(100)+chr(117)+chr(112))}")' "$jid"
    done
    ;;
  profile)
    request "${base}/jobs/${2:?usage: okc_api_retry.sh profile JOB_ID}" |
      python3 -c 'import json, re, sys
j=json.load(sys.stdin); r=j.get("result") or {}
print(f"job={j.get(chr(105)+chr(100))} status={j.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))} passed={r.get(chr(112)+chr(97)+chr(115)+chr(115)+chr(101)+chr(100))} score_us={r.get(chr(115)+chr(99)+chr(111)+chr(114)+chr(101)+chr(95)+chr(117)+chr(115))} speedup={r.get(chr(115)+chr(112)+chr(101)+chr(101)+chr(100)+chr(117)+chr(112))} ci={r.get(chr(115)+chr(112)+chr(101)+chr(101)+chr(100)+chr(117)+chr(112)+chr(95)+chr(99)+chr(105))} flags={r.get(chr(102)+chr(108)+chr(97)+chr(103)+chr(115))}")
for x in r.get("records", []):
 c=x.get("case", {})
 if x.get("event") == "benchmark": print(f"bench {x.get(chr(105)+chr(110)+chr(100)+chr(101)+chr(120)):2} {c.get(chr(108)+chr(111)+chr(97)+chr(100)):9} {c.get(chr(108)+chr(97)+chr(121)+chr(101)+chr(114)):7} B{c.get(chr(98)+chr(97)+chr(116)+chr(99)+chr(104)):2} L{c.get(chr(113)+chr(95)+chr(108)+chr(101)+chr(110),1)} median_us={x.get(chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)+chr(95)+chr(110)+chr(115),0)/1000:.3f}")
 if x.get("event") == "profile":
  m=re.search(r"total\s+([0-9.]+) ", x.get("report", "")); print(f"prof  {x.get(chr(105)+chr(110)+chr(100)+chr(101)+chr(120)):2} {c.get(chr(108)+chr(111)+chr(97)+chr(100)):9} {c.get(chr(108)+chr(97)+chr(121)+chr(101)+chr(114)):7} B{c.get(chr(98)+chr(97)+chr(116)+chr(99)+chr(104)):2} L{c.get(chr(113)+chr(95)+chr(108)+chr(101)+chr(110),1)} kernel_us={m.group(1) if m else chr(63)}")'
    ;;
  problem) request "${base}/problems/muse_attention_decode" ;;
  quota)
    request "${base}/problems/muse_attention_decode" |
      python3 -c 'import json, sys; print(json.load(sys.stdin)["limits"])'
    ;;
  leaderboard) request "${base}/leaderboard/muse_attention_decode" ;;
  overall) request "${base}/leaderboard" ;;
  rank)
    scope="${2:-problem}"
    if [[ "$scope" == "overall" ]]; then url="${base}/leaderboard"; else url="${base}/leaderboard/muse_attention_decode"; fi
    request "$url" |
      python3 -c 'import json, sys
team=sys.argv[1]; root=json.load(sys.stdin); hits=[]
def has_team(x): return isinstance(x,dict) and team in [str(v) for v in x.values() if isinstance(v,(str,int,float))]
def walk(x,path="root"):
 if isinstance(x,list):
  for i,v in enumerate(x):
   if has_team(v): hits.append({"rank":i+1,"total":len(x),"path":path,"row":v})
   walk(v,f"{path}[{i}]")
 elif isinstance(x,dict):
  for k,v in x.items(): walk(v,f"{path}.{k}")
walk(root)
for h in hits: print(json.dumps(h,sort_keys=True))
if not hits and has_team(root): print(json.dumps({"rank":None,"total":None,"path":"root","row":root},sort_keys=True))' "${3:-aditya3799}"
    ;;
  *)
    echo "usage: okc_api_retry.sh {check FILE|submit FILE|jobs|findhash HASH|near CREATED_AT [COUNT]|status JOB_ID|batch JOB_ID...|profile JOB_ID|problem|quota|leaderboard|overall|rank [problem|overall] [TEAM]}" >&2
    exit 2
    ;;
esac
