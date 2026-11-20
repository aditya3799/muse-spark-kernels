#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: direct_okc_submit.sh PROBLEM SOURCE [MODE]" >&2
  exit 2
fi

problem=$1
source_file=$2
mode=${3:-}
token=$(/home/aditya/.okc/bin/okc api token)
api=$(python3 -c 'import json; print(json.load(open("/home/aditya/.config/okc/config.json")).get("api") or "https://openkernelchallenge.dev/api/v1")')
payload=$(mktemp /tmp/okc-submit.XXXXXX.json)
trap 'rm -f -- "$payload"' EXIT

python3 -c 'import json,sys; payload={"code": open(sys.argv[1], encoding="utf-8").read()}; mode=sys.argv[2]; payload.update({"mode": mode} if mode else {}); json.dump(payload, open(sys.argv[3], "w", encoding="utf-8"), separators=(",", ":"))' "$source_file" "$mode" "$payload"

curl --ipv4 --http1.1 --fail-with-body --silent --show-error \
    --connect-timeout 20 --max-time 180 \
    -H "Authorization: Bearer $token" \
    -H 'Content-Type: application/json' \
    -H 'User-Agent: Go-http-client/1.1' \
    -H 'Connection: close' \
    --data-binary @"$payload" "$api/problems/$problem/submissions"
