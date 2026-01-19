import json
import os
import sys
import datetime
from tqdm import tqdm

# Add src path to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.rca_agent import run_rca_agent

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run RCA Agent evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases to run")
    args = parser.parse_args()

    # Path configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_file_path = os.path.join(base_dir, "data", "label.json")
    result_dir = os.path.join(base_dir, "result")
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Read label data
    print(f"Reading labels from {label_file_path}...")
    try:
        with open(label_file_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: Label file not found at {label_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse label file {label_file_path}")
        return

    if args.limit:
        labels = labels[:args.limit]
        print(f"Limiting execution to first {args.limit} cases.")

    evaluation_results = []
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(result_dir, f"evaluation_summary_{timestamp}.json")

    print(f"Starting evaluation on {len(labels)} cases...")
    
    # Iterate through cases
    for i, case in tqdm(enumerate(labels), total=len(labels), desc="Processing Cases"):
        start_time = case.get("start_time")
        end_time = case.get("end_time")
        
        if not start_time or not end_time:
            print(f"Skipping case {i}: Missing start_time or end_time")
            continue
            
        print(f"\nProcessing Case {i+1}/{len(labels)}: {start_time} - {end_time}")
        
        try:
            # Run the agent
            agent_result = run_rca_agent(start_time, end_time)
            
            # Combine ground truth with agent result
            combined_result = {
                "case_id": case.get("uuid", f"case_{i}"),
                "ground_truth": {
                    "fault_type": case.get("fault_type"),
                    "instance": case.get("instance"),
                    "start_time": start_time,
                    "end_time": end_time
                },
                "agent_result": agent_result
            }
            
            evaluation_results.append(combined_result)
            
            # Intermediate save (optional, good for long runs)
            if (i + 1) % 5 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
                    
        except Exception as e:
            print(f"Error processing case {i}: {str(e)}")
            evaluation_results.append({
                "case_id": case.get("uuid", f"case_{i}"),
                "error": str(e),
                "ground_truth": case
            })

    # Final save
    print(f"\nSaving final results to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
