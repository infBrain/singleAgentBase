import json
from typing import Any, Dict


def parse_rca_json_output(raw_text: str) -> Dict[str, Any]:
    cleaned_result = (raw_text or "").strip()

    json_block_start = cleaned_result.find("```json")
    if json_block_start != -1:
        block_content_start = json_block_start + len("```json")
        block_end = cleaned_result.find("```", block_content_start)
        if block_end != -1:
            cleaned_result = cleaned_result[block_content_start:block_end].strip()

    if "Answer:" in cleaned_result:
        cleaned_result = cleaned_result.split("Answer:")[-1].strip()

    if cleaned_result.startswith("```json"):
        cleaned_result = cleaned_result[7:]
    elif cleaned_result.startswith("```"):
        cleaned_result = cleaned_result[3:]

    if cleaned_result.endswith("```"):
        cleaned_result = cleaned_result[:-3]

    cleaned_result = cleaned_result.strip()

    try:
        return json.loads(cleaned_result)
    except json.JSONDecodeError:
        start_idx = cleaned_result.find("{")
        end_idx = cleaned_result.rfind("}")
        if start_idx != -1 and end_idx != -1:
            maybe_json = cleaned_result[start_idx:end_idx + 1]
            try:
                return json.loads(maybe_json)
            except Exception:
                pass

    return {
        "root_cause": "unknown",
        "raw_output": raw_text,
        "error": "Failed to parse LLM output as specific JSON format"
    }


if __name__ == "__main__":
        sample_output = """
Some irrelevant text before.
```json
{
    "anomaly type": "network",
    "root cause": [
        {"location": "node-1", "reason": "latency spike"}
    ]
}
```
Some irrelevant text after.
"""
        print(parse_rca_json_output(sample_output))
