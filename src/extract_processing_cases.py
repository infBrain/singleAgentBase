#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Tuple, Dict, Any


def _finalize_case(buffer: List[str]) -> Tuple[Dict[str, Any] | None, str | None]:
    raw_text = "".join(buffer).strip()
    if not raw_text:
        return None, "empty buffer"
    last_brace = raw_text.rfind("}")
    if last_brace != -1:
        raw_text = raw_text[: last_brace + 1]
    try:
        return json.loads(raw_text), None
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"


def extract_processing_cases(log_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    cases: List[Dict[str, Any]] = []
    errors: List[str] = []
    capturing = False
    buffer: List[str] = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not capturing:
                if "Processing Cases" in line:
                    capturing = True
                    brace_index = line.find("{")
                    buffer = [line[brace_index:]] if brace_index != -1 else []
                continue

            if "[HUMAN]" in line:
                case, error = _finalize_case(buffer)
                if case is not None:
                    cases.append(case)
                elif error:
                    errors.append(f"line {line_no}: {error}")
                capturing = False
                buffer = []
                continue

            buffer.append(line)

    if capturing:
        case, error = _finalize_case(buffer)
        if case is not None:
            cases.append(case)
        elif error:
            errors.append(f"eof: {error}")

    return cases, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract JSON cases between 'Processing Cases' and '[HUMAN]' markers."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the log file (e.g. result/mcp_result_0127.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="processing_cases.json",
        help="Output JSON path (default: processing_cases.json)",
    )
    parser.add_argument(
        "--errors",
        default="",
        help="Optional path to write extraction errors as JSONL",
    )

    args = parser.parse_args()

    cases, errors = extract_processing_cases(args.input)

    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(cases, out, ensure_ascii=False, indent=2)

    if args.errors:
        with open(args.errors, "w", encoding="utf-8") as err_out:
            for error in errors:
                err_out.write(json.dumps({"error": error}, ensure_ascii=False) + "\n")

    print(f"Extracted {len(cases)} cases -> {args.output}")
    if errors:
        print(f"Warnings: {len(errors)} malformed cases", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
