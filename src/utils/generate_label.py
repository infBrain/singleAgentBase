import json
import os
import argparse

def convert_jsonl_to_json(input_file, output_file):
    """
    Reads a JSONL file and writes it as a JSON array to the output file.
    """
    data_list = []
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Converted {len(data_list)} records.")
    
    print(f"Writing to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print("Done.")
    except Exception as e:
        print(f"Error writing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert groundtruth.jsonl to label.json format.")
    
    # Default paths are relative to the project root assuming the script is run from there
    # or adjusted based on where the user usually runs scripts.
    # Given the workspace info, 'data' and 'server' are in the root.
    
    default_input = os.path.join("data", "groundtruth.jsonl")
    default_output = os.path.join("data", "label.json") # A reasonable default based on context

    parser.add_argument("-i", "--input", type=str, default=default_input, help=f"Path to input JSONL file (default: {default_input})")
    parser.add_argument("-o", "--output", type=str, default=default_output, help=f"Path to output JSON file (default: {default_output})")
    
    args = parser.parse_args()
    
    convert_jsonl_to_json(args.input, args.output)
