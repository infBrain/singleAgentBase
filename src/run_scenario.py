import datetime
import json
from json import tool
import os
import sys
import asyncio
from typing import Dict, Any
from unittest import result

# Add src to sys.path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools import traditional_tools, mcp_tools
from src.utils.common_utils import _beijing_to_unix_seconds, _convert_to_beijing
from src.utils.multimodal_data import query_system_information

from src.agent.traditional_agent import run_traditional_agent
from src.agent.mcp_agent_call import run_mcp_agent


SCENARIO_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "scenario_tasks.json"
)

SCENARIO_JSON_RESULT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "result", f"scenario_tasks_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

SYSTEM_PROMPT = """
You are a query executor. Your job is to return ONLY the final results.

Rules:
- Do NOT output any analysis, explanation, reasoning, or summaries.
- Do NOT describe what you did. No bullet points.
- Output MUST be ONLY the raw results.

Output format:
- If the result is a time series, output should have pod/service/node name with values, like adservice-0:[value1, value2, value3, ...].
- If the result is grouped/top-k, output a list of {key, value}.
- If the result is knowledge/relations, output only the final list/path set.
- If no data, output: NO_DATA
- If error, output: ERROR
"""

TRADITIONAL_TOOLS = [
    traditional_tools.get_logs,
    traditional_tools.get_metrics,
    traditional_tools.get_traces,
    traditional_tools.detect_metrics,
    traditional_tools.detect_traces,
    traditional_tools.detect_logs,
    traditional_tools.get_system_info,
]

def load_scenarios(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_traditional_agent_scenario(scenario: Dict[str, Any], idx: int = 0):
    if scenario.get("start_time") and scenario.get("end_time"):
        start_time = scenario.get("start_time", "")
        end_time = scenario.get("end_time", "")
        prompt = (
            scenario["prompt"]
            .replace("${start_time}", start_time)
            .replace("${end_time}", end_time)
        )
    else:
        prompt = scenario["prompt"]
    print("[Traditional Agent] Running scenario:", idx)

    knowledge = query_system_information()

    result = run_traditional_agent(
        system_prompt=SYSTEM_PROMPT + "\n" + knowledge,
        user_prompt=prompt,
        tools=TRADITIONAL_TOOLS,
    )
    print("[Traditional Agent Result]:\n", result)
    return result


async def run_mcp_agent_async(scenario: Dict[str, Any], idx: int = 0):
    delay = 201 * 24 * 60
    if scenario.get("start_time") and scenario.get("end_time"):
        start_time = _beijing_to_unix_seconds(
            _convert_to_beijing(scenario.get("start_time", ""), delay=delay)
        )
        end_time = _beijing_to_unix_seconds(
            _convert_to_beijing(scenario.get("end_time", ""), delay=delay)
        )
        prompt = (
            scenario["prompt"]
            .replace("${start_time}", str(start_time))
            .replace("${end_time}", str(end_time))
        )
    else:
        prompt = scenario["prompt"]
    print("[MCP Agent] Running scenario:", idx)

    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "entity_info.json"), "r", encoding="utf-8") as f:
        entity_info = json.load(f)

    system_prompt = SYSTEM_PROMPT + "\n Your system entities information: \n" + json.dumps(entity_info)

    project_details = mcp_tools.build_project_details(
        workspace="zy-aiops-challenges-2025",
        region="cn-heyuan",
        sls_project="default-cms-1102382765107602-cn-heyuan",
        logstore="aiops-dataset-logs",
        metircstore="aiops-dataset-metrics",
        tracestore="aiops-dataset-traces",
    )

    result = await run_mcp_agent(
        system_prompt=system_prompt,
        project_details=project_details,
        user_prompt=prompt,
    )
    
    print("[MCP Agent Result]:\n", result)
    return result


if __name__ == "__main__":
    scenarios = load_scenarios(SCENARIO_JSON)
    for idx, scenario in enumerate(scenarios[:1]): 
        print(f"\n===== Scenario {idx} =====")
        print(f"Prompt: {scenario['prompt']}")
        print("--- [Traditional Agent] ---")
        result_traditional = run_traditional_agent_scenario(scenario, idx)
        print("--- [MCP Agent] ---")
        try:
            result_mcp = asyncio.run(run_mcp_agent_async(scenario, idx))
        except Exception as e:
            result_mcp = ""
            print(f"MCP agent failed: {e}")
        print("==========================\n")

        scenario["traditional_agent_result"] = result_traditional
        scenario["mcp_agent_result"] = result_mcp

    # Optionally save results back to a file
    with open(SCENARIO_JSON_RESULT, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=4, ensure_ascii=False)
    print(f"All scenario results saved to {SCENARIO_JSON_RESULT}")
