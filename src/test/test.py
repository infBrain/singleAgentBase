

import json
import os
import sys
# Add src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'result', '1-mcp_results_20260130_102159.jsonl')

results_cp = []

with open(file_path, 'r', encoding='utf-8') as f:
    buffer = ""
    for line in f:
        line = line.strip()
        if not line:
            continue
        buffer += line
        if line.endswith("}"):
            results_cp.append(json.loads(buffer))
            buffer = ""


with open(file_path.replace('.jsonl', '_modified.jsonl'), 'w', encoding='utf-8') as f:
    json.dump(results_cp, f, ensure_ascii=False, indent=2)