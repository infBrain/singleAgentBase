import asyncio
from math import log
import os
from re import A
import sys
import argparse
import json
import datetime

from openai import project
from tqdm import tqdm

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.mcp_rca_agent import run_mcp_agent
from src.agent.rca_agent import run_rca_agent
from src.tools.rca_output import parse_rca_json_output
from src.tools.utils import _convert_to_beijing, _beijing_to_unix_seconds


def build_system_prompt(start_time, end_time):
    return f"""You are a Site Reliability Engineer (SRE) agent responsible for Root Cause Analysis (RCA).
Your task is to determine the anomaly type and root cause of the fault that occurred between {start_time} and {end_time}.
You have access to various tools to help you investigate metrics, traces, logs, and system information.

The root cause **must** be specific instance name (pod e.g. adservice-0, service e.g. adservice, node e.g. aiops-k8s-01) without any other information, and should be returned in the following JSON format (no more than three):

{{
  "anomaly type": "<anomaly type>",
  "root cause": [
    {{"location": "<instance_name>", "reason": "<simple explanation>"}},
    {{"location": "<instance_name>", "reason": "<simple explanation>"}},
    ...
  ]
}}

🔧 **Analysis Steps — Please follow carefully:**
1. If a tool exists for anomaly type classification, use it first to identify the anomaly category.
2. Within the given anomaly time range, you **must** perform anomaly detection on all three: time series (metrics), logs, and traces. Do not skip any of these checks.
3. Synthesize observations from the three sources, identify candidate entities.
4. Retrieve the system topology and call graph. Validating the fault propagation path (e.g., A calls B, and B is slow) is crucial for distinguishing root causes from symptoms.
5. Validate candidates by checking upstream/downstream impact paths and consistency across signals.
6. Before answering, explicitly judge the most likely root cause based on evidence strength and consistency.
7. Return the final result strictly in the required JSON format.

🔍 **Root Cause Localization Steps:**
1. List the top suspicious entities (pod/service/node) based on anomalies observed.
2. Use system topology information to cross-check upstream/downstream relationships and validate propagation paths.
3. Select the most likely root cause instance(s) and provide concise evidence for each.

🧭 **Reasoning Guidance:**
- Prefer evidence-driven conclusions; do not guess without supporting signals.
- If multiple candidates exist, list up to three with concise reasons.
- If data is missing or inconclusive, state "Unknown" for anomaly type and provide an empty root_cause list or explain unknown in reasons.
- Be adaptive: if a check is inconclusive, try an alternative signal or a narrower scope, then re-evaluate.

⚠️ **Important:**
- Think step by step, justify your actions, and always use the tools logically and effectively to pinpoint the root cause.
- If a pod is the root cause (e.g. adservice-0), the corresponding service (e.g. adservice) might also be the root cause!
- If you find no anomalies in one tool, move to the next.
- Combine the insights from multiple tools to form a robust conclusion.
- If you cannot determine the root cause, honestly state root cause unknown in your final answer.

## Final Answer Format

When you have sufficient information to answer the question, you **MUST** provide the final answer as a valid JSON object strictly following the format above.
Do **NOT** wrap the JSON in markdown code blocks (like ```json ... ```).
Do **NOT** add any text before or after the JSON.
Do **NOT** include tool call traces or any intermediate reasoning in the final answer.
Your final response must be **only** the JSON object.
Just output the raw JSON string.
"""


def build_user_message(start_time, end_time):
    return f"A fault occurred from  {start_time} to {end_time}. please locate the issue root cause"


def build_project_details(
    workspace, region, project, logstore, metircstore, tracestore
):
    return f"""Your UModel workspace is '{workspace}' in region '{region}', and the SLS project is '{project}'.
    The logstore is '{logstore}', the metricstore is '{metircstore}', the tracestore is '{tracestore}'.
    Use this information when configuring your data source connections.
    Only use metric_set values from the list returned by umodel_list_metric_sets. Do not invent metric_set names.
    """


## MCP Agent Execution
async def run_mcp_only(
    start_time,
    end_time,
    sls_endpoints="cn-heyuan=cn-heyuan.log.aliyuncs.com",
    cms_endpoints="cn-heyuan=cms.cn-heyuan.aliyuncs.com",
    ground_truth=None,
    uuid=None,
    delay=0,
):
    prompt_start_time = _beijing_to_unix_seconds(
        _convert_to_beijing(start_time, delay=201 * 24 * 60)
    )
    prompt_end_time = _beijing_to_unix_seconds(
        _convert_to_beijing(end_time, delay=201 * 24 * 60)
    )
    system_prompt = build_system_prompt(prompt_start_time, prompt_end_time)
    user_message = build_user_message(prompt_start_time, prompt_end_time)
    project_details = build_project_details(
        workspace="default-cms-1102382765107602-cn-heyuan",
        region="cn-heyuan",
        project="default-cms-1102382765107602-cn-heyuan",
        logstore="aiops-dataset-logs",
        metircstore="aiops-dataset-metrics",
        tracestore="aiops-dataset-traces",
    )
    # mcp_query = f"{system_prompt}\n{project_details}\nUser Request:\n{user_message}\n"

    # python_executable = sys.executable  # stdio mode need python executable

    # access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    # access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    # mcp_result_text = await run_mcp_agent(
    #     query=mcp_query,
    #     connection_mode="stdio",
    #     url_or_cmd=python_executable,
    #     access_key_id=access_key_id,
    #     access_key_secret=access_key_secret,
    #     # sls_endpoints=sls_endpoints if sls_endpoints else "cn-heyuan=default-cms-1102382765107602-cn-heyuan",
    #     # cms_endpoints=cms_endpoints if cms_endpoints else "cn-heyuan=default-cms-1102382765107602-cn-heyuan",
    #     sls_endpoints=sls_endpoints,
    #     cms_endpoints=cms_endpoints,
    # )
    mcp_result_text = await run_mcp_agent(
        system_prompt=system_prompt,
        project_details=project_details,
        user_prompt=user_message,
        connection_mode="sse",
        url="http://127.0.0.1:8000/sse",
    )

    mcp_result = parse_rca_json_output(mcp_result_text)
    if uuid:
        mcp_result["uuid"] = uuid
    mcp_result["start_time"] = start_time
    mcp_result["end_time"] = end_time
    if ground_truth:
        mcp_result["ground_truth"] = ground_truth
    return mcp_result


## Local RCA Agent Execution
def run_rca_only(
    start_time, end_time, uuid=None, instance_type=None, ground_truth=None
):
    system_prompt = build_system_prompt(start_time, end_time)
    user_message = build_user_message(start_time, end_time)
    rca_result = run_rca_agent(start_time, end_time, system_prompt, user_message)
    if uuid:
        rca_result["uuid"] = uuid
    if instance_type:
        rca_result["instance_type"] = instance_type
    rca_result["start_time"] = start_time
    rca_result["end_time"] = end_time
    if ground_truth:
        rca_result["ground_truth"] = ground_truth
    return rca_result


# async def run_comparison(
#     workspace,
#     region,
#     project,
#     start_time,
#     end_time,
#     sls_endpoints=None,
#     cms_endpoints=None,
# ):
#     print("=" * 60)
#     print("STARTING AGENT COMPARISON")
#     print("=" * 60)
#     print(f"Time Range: {start_time} to {end_time}")
#     print("=" * 60)

#     # --- 1. Run MCP Agent ---
#     print("\n" + "-" * 20 + " Running MCP Agent " + "-" * 20 + "\n")
#     mcp_result = await run_mcp_only(
#         start_time=start_time,
#         end_time=end_time,
#         sls_endpoints=sls_endpoints,
#         cms_endpoints=cms_endpoints,
#     )

#     # --- 2. Run Local RCA Agent ---
#     print("\n" + "-" * 20 + " Running Local RCA Agent " + "-" * 20 + "\n")
#     try:
#         rca_result = run_rca_only(
#             start_time=start_time,
#             end_time=end_time,
#         )
#     except Exception as e:
#         rca_result = {"error": str(e)}

#     # --- 3. Compare Results ---
#     print("\n" + "=" * 60)
#     print("COMPARISON RESULT")
#     print("=" * 60)

#     print("\n--- MCP Agent Output (JSON) ---")
#     print(json.dumps(mcp_result, indent=2, ensure_ascii=False))

#     print("\n--- Local RCA Agent Output (JSON) ---")
#     print(json.dumps(rca_result, indent=2, ensure_ascii=False))

#     # Save comparison to file
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     result_dir = os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result"
#     )
#     if not os.path.exists(result_dir):
#         os.makedirs(result_dir)

#     filename = f"comparison_{timestamp}.txt"
#     filepath = os.path.join(result_dir, filename)

#     with open(filepath, "w", encoding="utf-8") as f:
#         f.write("=== MCP Agent Output ===\n")
#         f.write(json.dumps(mcp_result, indent=2, ensure_ascii=False) + "\n\n")
#         f.write("=== Local RCA Agent Output ===\n")
#         f.write(json.dumps(rca_result, indent=2, ensure_ascii=False) + "\n")

#     print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MCP Agent and Local RCA Agent"
    )
    # parser.add_argument("--start-time", default="2025-06-05T16:10:02Z", help="Start time in ISO format")
    # parser.add_argument("--end-time", default="2025-06-05T16:31:02Z", help="End time in ISO format")
    # parser.add_argument("--task", default="please locate the issue root cause", help="Description of the problem")
    # parser.add_argument(
    #     "--sls-endpoints", help="Override SLS endpoints (e.g. 'cn-region=host')"
    # )
    # parser.add_argument(
    #     "--cms-endpoints", help="Override CMS endpoints (e.g. 'cn-region=host')"
    # )
    parser.add_argument(
        "--mode",
        choices=["mcp", "rca", "both"],
        default="mcp",
        # default="rca",
        help="Run MCP agent, local RCA agent, or both",
    )

    args = parser.parse_args()
    result_answers = []
    try:
        with open(os.path.join("data", "label_test.json"), "r", encoding="utf-8") as f:
            labels = json.load(f)
            labels = labels[:1]  # For testing, process only the first case

        for case in tqdm(labels, desc="Processing Cases", total=len(labels)):
            start_time = case["start_time"]
            end_time = case["end_time"]

            if args.mode == "mcp":
                result = asyncio.run(
                    run_mcp_only(
                        uuid=case.get("uuid"),
                        start_time=start_time,
                        end_time=end_time,
                        ground_truth=case.get("instance"),
                    )
                )
                result_answers.append(json.dumps(result, indent=2, ensure_ascii=False))
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif args.mode == "rca":
                result = run_rca_only(
                    uuid=case.get("uuid"),
                    start_time=start_time,
                    end_time=end_time,
                    instance_type=case.get("instance_type"),
                    ground_truth=case.get("instance"),
                )
                result_answers.append(json.dumps(result, indent=2, ensure_ascii=False))
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                # asyncio.run(run_comparison(
                #     workspace=args.workspace,
                #     region=args.region,
                #     project=args.project,
                #     start_time=args.start_time,
                #     end_time=args.end_time,
                #     sls_endpoints=args.sls_endpoints,
                #     cms_endpoints=args.cms_endpoints
                # ))
                pass

        # Save all results to a single file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result"
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        filename = f"{args.mode}_results_{timestamp}.jsonl"
        filepath = os.path.join(result_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for answer in result_answers:
                f.write(answer + "\n\n")
        print(f"\nAll results saved to {filepath}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
