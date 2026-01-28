#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Iterable, List


def _load_mcp_items(mcp_json_path: str) -> List[Dict]:
    with open(mcp_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("MCP file must contain a JSON array")
    return [item for item in data if isinstance(item, dict) and item.get("uuid")]


def _iter_jsonl(path: str) -> Iterable[Dict]:
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    index = 0
    length = len(content)
    while index < length:
        while index < length and content[index].isspace():
            index += 1
        if index >= length:
            break
        obj, end = decoder.raw_decode(content, index)
        yield obj
        index = end


def align_by_uuid(mcp_json_path: str, rca_jsonl_path: str) -> tuple[List[Dict], List[Dict]]:
    """Filter MCP JSON and RCA JSONL entries by shared UUIDs."""
    mcp_items = _load_mcp_items(mcp_json_path)
    mcp_map = {item["uuid"]: item for item in mcp_items}
    matched_rca: List[Dict] = []
    matched_uuids: set[str] = set()
    for entry in _iter_jsonl(rca_jsonl_path):
        uuid = entry.get("uuid")
        if uuid in mcp_map:
            matched_rca.append(entry)
            matched_uuids.add(uuid)
    matched_mcp = [mcp_map[uuid] for uuid in matched_uuids if uuid in mcp_map]
    return matched_mcp, matched_rca


def _default_output_paths(mcp_json_path: str, rca_jsonl_path: str) -> tuple[str, str]:
    mcp_base = os.path.basename(mcp_json_path)
    mcp_stem = os.path.splitext(mcp_base)[0]
    rca_base = os.path.basename(rca_jsonl_path)
    rca_stem = os.path.splitext(rca_base)[0]
    base_dir = os.path.join("eval_result", "eval_mcp")
    return (
        os.path.join(base_dir, f"mcp_{mcp_stem}.json"),
        os.path.join(base_dir, f"rca_{rca_stem}.json"),
    )


def main() -> int:
    # python src/utils/align_results.py --mcp result/mcp_processing_result_0127.json --rca result/rca_results_20260125_063144.jsonl
    parser = argparse.ArgumentParser(
        description="Align MCP JSON and RCA JSONL by UUID; output matched RCA entries."
    )
    parser.add_argument("--mcp", required=True, help="Path to MCP JSON array file")
    parser.add_argument("--rca", required=True, help="Path to RCA JSONL file")
    parser.add_argument(
        "--output-mcp",
        default="",
        help="Output MCP JSON path (default: eval_result/eval_mcp/mcp_<mcp_name>.json)",
    )
    parser.add_argument(
        "--output-rca",
        default="",
        help="Output RCA JSON path (default: eval_result/eval_mcp/rca_<rca_name>.json)",
    )

    args = parser.parse_args()
    default_mcp_path, default_rca_path = _default_output_paths(args.mcp, args.rca)
    output_mcp = args.output_mcp or default_mcp_path
    output_rca = args.output_rca or default_rca_path
    os.makedirs(os.path.dirname(output_mcp), exist_ok=True)
    os.makedirs(os.path.dirname(output_rca), exist_ok=True)

    matched_mcp, matched_rca = align_by_uuid(args.mcp, args.rca)
    with open(output_mcp, "w", encoding="utf-8") as out_mcp:
        json.dump(matched_mcp, out_mcp, ensure_ascii=False, indent=2)
    with open(output_rca, "w", encoding="utf-8") as out_rca:
        json.dump(matched_rca, out_rca, ensure_ascii=False, indent=2)

    print(
        f"Matched {len(matched_mcp)} MCP entries -> {output_mcp}; "
        f"{len(matched_rca)} RCA entries -> {output_rca}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
