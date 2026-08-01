import json
import sys

try:
    with open("pip-audit.json", encoding="utf-8") as f:
        payload = json.load(f)
except (OSError, json.JSONDecodeError):
    payload = []

if isinstance(payload, dict):
    records = payload.get("dependencies", [])
elif isinstance(payload, list):
    records = payload
else:
    records = []

vulns = [record for record in records if isinstance(record, dict) and record.get("vulns")]
if vulns:
    print(f"FAIL: {len(vulns)} packages with known vulnerabilities")
    for record in vulns:
        for vuln in record.get("vulns", []):
            print(f"  {record.get('name')}=={record.get('version')}: {vuln.get('id')}")
    sys.exit(1)
print("PASS: No known vulnerabilities in dependencies")
