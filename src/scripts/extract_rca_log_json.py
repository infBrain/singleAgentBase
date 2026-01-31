#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


# =========================
# Schema definition
# =========================

REQUIRED_KEYS = {
    "anomaly type",
    "root cause",
    "uuid",
    "instance_type",
    "start_time",
    "end_time",
    "ground_truth",
}


# =========================
# Core extraction logic
# =========================

def extract_json_objects_from_log(text: str) -> List[Dict[str, Any]]:
    """
    Extract all valid JSON objects from mixed log text.
    Supports multi-line and consecutive JSON blocks.
    """
    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    results = []

    while idx < length:
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break

        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                results.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1

    return results


def is_final_anomaly_result(obj: Dict[str, Any]) -> bool:
    """
    Check whether the JSON object matches the final anomaly result schema.
    """
    return REQUIRED_KEYS.issubset(obj.keys())


def extract_final_anomaly_results_from_log(text: str) -> List[Dict[str, Any]]:
    """
    Extract only final anomaly result JSON objects.
    """
    all_jsons = extract_json_objects_from_log(text)
    return [obj for obj in all_jsons if is_final_anomaly_result(obj)]


# =========================
# Main entry
# =========================

def main(input_path: str, output_path: str):
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input log file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    results = extract_final_anomaly_results_from_log(log_text)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("✅ Extraction finished")
    print(f"📄 Input log  : {input_path}")
    print(f"📦 Results   : {len(results)} items")
    print(f"💾 Output    : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract final anomaly result JSONs from agent logs"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input log file (e.g. result/agent.log)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSON file (e.g. result/anomaly_results.json)",
    )

    args = parser.parse_args()
    main(args.input, args.output)
