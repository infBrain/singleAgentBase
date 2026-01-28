#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Iterable, List, Tuple


def _iter_json_objects(text: str) -> Iterable[Dict]:
    """Iterate JSON objects from a raw text blob (JSONL or concatenated JSON)."""
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        obj, end = decoder.raw_decode(text, index)
        yield obj
        index = end


def _load_json_any(path: str) -> List[Dict]:
    """Load JSON from a file; supports JSON array, single JSON, or JSONL."""
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    return [item for item in _iter_json_objects(content) if isinstance(item, dict)]


def _extract_locations(entry: Dict) -> List[str]:
    """Extract root-cause locations from an entry."""
    root_cause = entry.get("root cause") or entry.get("root_cause")
    if isinstance(root_cause, list):
        locations = []
        for item in root_cause:
            if isinstance(item, dict) and item.get("location"):
                locations.append(str(item["location"]))
        return locations
    return []


def _normalize_ground_truth(value) -> List[str]:
    """Normalize ground_truth to a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    """De-duplicate while preserving original order."""
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _hit_at_k(locations: List[str], ground_set: set, k: int) -> int:
    """Hit@K: whether any ground truth appears in top-K locations."""
    if not ground_set:
        return 0
    topk = locations[:k]
    return 1 if any(loc in ground_set for loc in topk) else 0


def _precision_recall_f1(locations: List[str], ground_set: set) -> Tuple[float, float, float]:
    """Set-based precision/recall/F1 between predicted locations and ground truth."""
    if not ground_set and not locations:
        return 0.0, 0.0, 0.0
    loc_set = set(locations)
    true_pos = len(loc_set & ground_set)
    precision = true_pos / len(loc_set) if loc_set else 0.0
    recall = true_pos / len(ground_set) if ground_set else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _mrr(locations: List[str], ground_set: set) -> float:
    """Mean Reciprocal Rank for the first correct location."""
    if not ground_set:
        return 0.0
    for idx, loc in enumerate(locations, start=1):
        if loc in ground_set:
            return 1.0 / idx
    return 0.0


def _average_precision(locations: List[str], ground_set: set) -> float:
    """Average Precision for ranked locations."""
    if not ground_set:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for idx, loc in enumerate(locations, start=1):
        if loc in ground_set:
            hit_count += 1
            precision_sum += hit_count / idx
    return precision_sum / len(ground_set) if ground_set else 0.0


def _ndcg(locations: List[str], ground_set: set) -> float:
    """Normalized DCG for ranked locations."""
    if not ground_set:
        return 0.0
    dcg = 0.0
    for idx, loc in enumerate(locations, start=1):
        if loc in ground_set:
            dcg += 1.0 / (1.0 + (idx - 1))
    ideal_hits = min(len(ground_set), len(locations))
    idcg = sum(1.0 / (1.0 + i) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def score_dataset(items: List[Dict], name: str) -> Dict:
    """Compute multiple ranking and set-based metrics for one dataset."""
    total = len(items)
    stats = {
        "total": total,
        "with_ground_truth": 0,
        "with_locations": 0,
        "hit_at_any": 0,
        "hit_at_1": 0,
        "hit_at_3": 0,
        "hit_at_5": 0,
    }

    misses: List[Dict] = []

    for entry in items:
        # Per-case hit statistics.
        ground_truth = _normalize_ground_truth(entry.get("ground_truth"))
        ground_set = {item for item in ground_truth if item}
        if ground_set:
            stats["with_ground_truth"] += 1

        locations = _dedupe_keep_order(_extract_locations(entry))
        if locations:
            stats["with_locations"] += 1

        hit_any = bool(ground_set and any(loc in ground_set for loc in locations))
        if hit_any:
            stats["hit_at_any"] += 1
        stats["hit_at_1"] += _hit_at_k(locations, ground_set, 1)
        stats["hit_at_3"] += _hit_at_k(locations, ground_set, 3)
        stats["hit_at_5"] += _hit_at_k(locations, ground_set, 5)
        if ground_set and not hit_any:
            misses.append(
                {
                    "uuid": entry.get("uuid"),
                    "predicted_locations": locations,
                    "ground_truth": ground_truth,
                }
            )

    precision_list: List[float] = []
    recall_list: List[float] = []
    f1_list: List[float] = []
    mrr_list: List[float] = []
    map_list: List[float] = []
    ndcg_list: List[float] = []

    for entry in items:
        # Per-case ranking metrics.
        ground_truth = _normalize_ground_truth(entry.get("ground_truth"))
        ground_set = {item for item in ground_truth if item}
        locations = _dedupe_keep_order(_extract_locations(entry))
        precision, recall, f1 = _precision_recall_f1(locations, ground_set)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        mrr_list.append(_mrr(locations, ground_set))
        map_list.append(_average_precision(locations, ground_set))
        ndcg_list.append(_ndcg(locations, ground_set))

    rates = {
        "hit_at_any_rate": stats["hit_at_any"] / stats["with_ground_truth"]
        if stats["with_ground_truth"]
        else 0,
        "coverage_rate": stats["with_locations"] / total if total else 0,
        "hit_at_1_rate": stats["hit_at_1"] / stats["with_ground_truth"]
        if stats["with_ground_truth"]
        else 0,
        "hit_at_3_rate": stats["hit_at_3"] / stats["with_ground_truth"]
        if stats["with_ground_truth"]
        else 0,
        "hit_at_5_rate": stats["hit_at_5"] / stats["with_ground_truth"]
        if stats["with_ground_truth"]
        else 0,
        "precision": sum(precision_list) / total if total else 0,
        "recall": sum(recall_list) / total if total else 0,
        "f1": sum(f1_list) / total if total else 0,
        "mrr": sum(mrr_list) / total if total else 0,
        "map": sum(map_list) / total if total else 0,
        "ndcg": sum(ndcg_list) / total if total else 0,
    }

    summary = {
        "name": name,
        "stats": stats,
        "rates": rates,
        "misses": misses,
    }
    return summary


def _default_output_path(mcp_path: str, rca_path: str) -> str:
    """Build default output path under eval_result/eval_mcp."""
    mcp_stem = os.path.splitext(os.path.basename(mcp_path))[0]
    rca_stem = os.path.splitext(os.path.basename(rca_path))[0]
    return os.path.join("eval_result", f"eval_{mcp_stem}_and_{rca_stem}.json")


def main() -> int:
    """CLI entrypoint for scoring MCP and RCA results separately."""
    parser = argparse.ArgumentParser(description="Score MCP and RCA results separately.")
    parser.add_argument("--mcp", required=True, help="Path to MCP JSON file")
    parser.add_argument("--rca", required=True, help="Path to RCA JSON or JSONL file")
    parser.add_argument("--output", default="", help="Output JSON path")

    args = parser.parse_args()
    mcp_items = _load_json_any(args.mcp)
    rca_items = _load_json_any(args.rca)

    mcp_summary = score_dataset(mcp_items, "mcp")
    rca_summary = score_dataset(rca_items, "rca")
    output_path = args.output or _default_output_path(args.mcp, args.rca)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "mcp": mcp_summary,
        "rca": rca_summary,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Evaluation written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
