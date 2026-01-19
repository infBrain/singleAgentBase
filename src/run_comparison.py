import asyncio
import os
from re import A
import sys
import argparse
import json
import datetime

from tqdm import tqdm

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.mcp_rca_agent import run_agent as run_mcp_agent
from src.agent.rca_agent import run_rca_agent
from src.tools.rca_output import parse_rca_json_output


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

The system information you can use includes:

**All Nodes**:
['aiops-k8s-01', 'aiops-k8s-02', 'aiops-k8s-03', 'aiops-k8s-04', 'aiops-k8s-05', 'aiops-k8s-06','aiops-k8s-07', 'aiops-k8s-08', 'k8s-master1', 'k8s-master2', 'k8s-master3']

**All services**:
['adservice', 'cartservice', 'currencyservice', 'productcatalogservice', 'checkoutservice','recommendationservice', 'shippingservice','emailservice', 'paymentservice']
Each service has three pods, e.g. adservice has three pods adservice-0, adservice-1, adservice-2

**Relationship between Services**:
A->B means service A call service B.
frontend->adservice,frontend->productcatalogservice,frontend->currencyservice,frontend->recommendationservice,cartservice,checkoutservice->paymentservice->shippingservice->emailservice


**All metrics**:
apm_metric_names = ["client_error_ratio","error_ratio","request","response","rrt","server_error_ratio","timeout"]
infra_pod_metric_names = ["pod_cpu_usage", "pod_fs_reads_bytes", "pod_fs_writes_bytes", "pod_network_receive_bytes", "pod_network_receive_packets", "pod_network_transmit_bytes", "pod_network_transmit_packets", "pod_processes"]
infra_node_metric_names = ["node_cpu_usage_rate", "node_filesystem_usage_rate", "node_memory_usage_rate", "node_network_receive_packets_total", "node_network_transmit_packets_total", "node_sockstat_TCP_inuse"]


**Kubernetes info**
NAME                            READY   STATUS      RESTARTS        AGE     IP              NODE           NOMINATED NODE   READINESS GATES
adservice-0                     1/1     Running     8 (4m14s ago)   7d5h    10.233.89.223   aiops-k8s-08   <none>           <none>
adservice-1                     1/1     Running     12 (14m ago)    7d5h    10.233.81.32    aiops-k8s-03   <none>           <none>
adservice-2                     1/1     Running     5 (63m ago)     7d5h    10.233.85.74    aiops-k8s-07   <none>           <none>
cartservice-0                   1/1     Running     2 (14h ago)     7d5h    10.233.81.73    aiops-k8s-03   <none>           <none>
cartservice-1                   1/1     Running     2 (14h ago)     7d5h    10.233.78.180   aiops-k8s-01   <none>           <none>
cartservice-2                   1/1     Running     2 (14h ago)     7d5h    10.233.77.46    aiops-k8s-04   <none>           <none>
checkoutservice-0               1/1     Running     1 (3d ago)      7d5h    10.233.85.52    aiops-k8s-07   <none>           <none>
checkoutservice-1               1/1     Running     1 (3d ago)      7d5h    10.233.81.234   aiops-k8s-03   <none>           <none>
checkoutservice-2               1/1     Running     1 (3d ago)      7d5h    10.233.77.2     aiops-k8s-04   <none>           <none>
currencyservice-0               1/1     Running     2 (7h3m ago)    2d22h   10.233.85.212   aiops-k8s-07   <none>           <none>
currencyservice-1               1/1     Running     2 (7h3m ago)    2d22h   10.233.81.6     aiops-k8s-03   <none>           <none>
currencyservice-2               1/1     Running     2 (7h3m ago)    2d22h   10.233.79.185   aiops-k8s-06   <none>           <none>
emailservice-0                  1/1     Running     4 (11h ago)     7d5h    10.233.85.75    aiops-k8s-07   <none>           <none>
emailservice-1                  1/1     Running     4 (11h ago)     7d5h    10.233.79.60    aiops-k8s-06   <none>           <none>
emailservice-2                  1/1     Running     4 (11h ago)     7d5h    10.233.78.139   aiops-k8s-01   <none>           <none>
example-ant-29107680-n6c8f      1/1     Running     0               15h     10.233.74.15    aiops-k8s-05   <none>           <none>
frontend-0                      1/1     Running     0               7d5h    10.233.85.77    aiops-k8s-07   <none>           <none>
frontend-1                      1/1     Running     0               7d5h    10.233.81.88    aiops-k8s-03   <none>           <none>
frontend-2                      1/1     Running     0               7d5h    10.233.74.117   aiops-k8s-05   <none>           <none>
paymentservice-0                1/1     Running     3 (19h ago)     7d5h    10.233.81.216   aiops-k8s-03   <none>           <none>
paymentservice-1                1/1     Running     3 (19h ago)     7d5h    10.233.89.213   aiops-k8s-08   <none>           <none>
paymentservice-2                1/1     Running     3 (19h ago)     7d5h    10.233.78.103   aiops-k8s-01   <none>           <none>
productcatalogservice-0         1/1     Running     1 (2d23h ago)   3d6h    10.233.74.105   aiops-k8s-05   <none>           <none>
productcatalogservice-1         1/1     Running     1 (2d23h ago)   3d6h    10.233.81.242   aiops-k8s-03   <none>           <none>
productcatalogservice-2         1/1     Running     1 (2d23h ago)   3d6h    10.233.85.21    aiops-k8s-07   <none>           <none>
recommendationservice-0         1/1     Running     1 (25h ago)     2d7h    10.233.85.42    aiops-k8s-07   <none>           <none>
recommendationservice-1         1/1     Running     1 (25h ago)     2d7h    10.233.81.146   aiops-k8s-03   <none>           <none>
recommendationservice-2         1/1     Running     1 (25h ago)     2d7h    10.233.89.86    aiops-k8s-08   <none>           <none>
redis-cart-0                    1/1     Running     0               7d5h    10.233.89.187   aiops-k8s-08   <none>           <none>
shippingservice-0               1/1     Running     0               2d19h   10.233.81.210   aiops-k8s-03   <none>           <none>
shippingservice-1               1/1     Running     0               2d19h   10.233.85.163   aiops-k8s-07   <none>           <none>
shippingservice-2               1/1     Running     0               2d19h   10.233.89.25    aiops-k8s-08   <none>           <none>


🔧 **Analysis Steps — Please follow carefully:**
1. If a tool exists for anomaly type classification (e.g., classifier), use it first to identify the anomaly category.
2. Within the given anomaly time range, you **must** perform anomaly detection on all three: time series (metrics), logs, and traces. Do not skip any of these checks.
3. Synthesize observations from the three sources, identify candidate entities, and correlate with dependency/topology context.
4. Validate candidates by checking upstream/downstream impact paths and consistency across signals.
5. Before answering, explicitly judge the most likely root cause based on evidence strength and consistency.
6. Return the final result strictly in the required JSON format.

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


async def run_mcp_only(start_time, end_time, sls_endpoints=None, cms_endpoints=None):
    system_prompt = build_system_prompt(start_time, end_time)
    user_message = build_user_message(start_time, end_time)
    mcp_query = f"{system_prompt}\n\nUser Request:\n{user_message}\n"

    python_executable = sys.executable
    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    mcp_result_text = await run_mcp_agent(
        query=mcp_query,
        connection_mode="stdio",
        url_or_cmd=python_executable,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        sls_endpoints=sls_endpoints,
        cms_endpoints=cms_endpoints,
    )

    mcp_result = parse_rca_json_output(mcp_result_text)
    mcp_result["start_time"] = start_time
    mcp_result["end_time"] = end_time
    return mcp_result


def run_rca_only(start_time, end_time, ground_truth=None):
    system_prompt = build_system_prompt(start_time, end_time)
    user_message = build_user_message(start_time, end_time)
    rca_result = run_rca_agent(start_time, end_time, system_prompt, user_message)
    rca_result["start_time"] = start_time
    rca_result["end_time"] = end_time
    if ground_truth:
        rca_result["ground_truth"] = ground_truth
    return rca_result


async def run_comparison(
    workspace,
    region,
    project,
    start_time,
    end_time,
    sls_endpoints=None,
    cms_endpoints=None,
):
    print("=" * 60)
    print("STARTING AGENT COMPARISON")
    print("=" * 60)
    print(f"Time Range: {start_time} to {end_time}")
    print("=" * 60)

    # --- 1. Run MCP Agent ---
    print("\n" + "-" * 20 + " Running MCP Agent " + "-" * 20 + "\n")
    mcp_result = await run_mcp_only(
        start_time=start_time,
        end_time=end_time,
        sls_endpoints=sls_endpoints,
        cms_endpoints=cms_endpoints,
    )

    # --- 2. Run Local RCA Agent ---
    print("\n" + "-" * 20 + " Running Local RCA Agent " + "-" * 20 + "\n")
    try:
        rca_result = run_rca_only(
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        rca_result = {"error": str(e)}

    # --- 3. Compare Results ---
    print("\n" + "=" * 60)
    print("COMPARISON RESULT")
    print("=" * 60)

    print("\n--- MCP Agent Output (JSON) ---")
    print(json.dumps(mcp_result, indent=2, ensure_ascii=False))

    print("\n--- Local RCA Agent Output (JSON) ---")
    print(json.dumps(rca_result, indent=2, ensure_ascii=False))

    # Save comparison to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result"
    )
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    filename = f"comparison_{timestamp}.txt"
    filepath = os.path.join(result_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== MCP Agent Output ===\n")
        f.write(json.dumps(mcp_result, indent=2, ensure_ascii=False) + "\n\n")
        f.write("=== Local RCA Agent Output ===\n")
        f.write(json.dumps(rca_result, indent=2, ensure_ascii=False) + "\n")

    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare MCP Agent and Local RCA Agent"
    )
    parser.add_argument(
        "--workspace", default="aiops-challenges-2025", help="UModel Workspace Name"
    )
    parser.add_argument("--region", default="cn-heyuan", help="Region ID")
    parser.add_argument(
        "--project",
        default="cms-aiops-challenges-2025-dbj7stfif6apnzrb",
        help="SLS Project Name",
    )
    # parser.add_argument("--start-time", default="2025-06-05T16:10:02Z", help="Start time in ISO format")
    # parser.add_argument("--end-time", default="2025-06-05T16:31:02Z", help="End time in ISO format")
    # parser.add_argument("--task", default="please locate the issue root cause", help="Description of the problem")
    parser.add_argument(
        "--sls-endpoints", help="Override SLS endpoints (e.g. 'cn-region=host')"
    )
    parser.add_argument(
        "--cms-endpoints", help="Override CMS endpoints (e.g. 'cn-region=host')"
    )
    parser.add_argument(
        "--mode",
        choices=["mcp", "rca", "both"],
        default="rca",
        help="Run MCP agent, local RCA agent, or both",
    )

    args = parser.parse_args()
    result_answers = []
    try:
        with open(os.path.join("data", "label.json"), "r", encoding="utf-8") as f:
            labels = json.load(f)
            # labels = labels[:1]  # For testing, process only the first case

        for case in tqdm(labels, desc="Processing Cases", total=len(labels)):
            start_time = case["start_time"]
            end_time = case["end_time"]

            if args.mode == "mcp":
                result = asyncio.run(
                    run_mcp_only(
                        start_time=start_time,
                        end_time=end_time,
                        sls_endpoints=args.sls_endpoints,
                        cms_endpoints=args.cms_endpoints,
                    )
                )
                result_answers.append(json.dumps(result, indent=2, ensure_ascii=False))
                print(json.dumps(result, indent=2, ensure_ascii=False))
            elif args.mode == "rca":
                result = run_rca_only(
                    start_time=start_time,
                    end_time=end_time,
                    ground_truth=case.get("service") if case.get("service") != "" else case.get("instance")
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
