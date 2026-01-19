from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
import os
import asyncio
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.kubernetes_tools import *
from tools.multimodal_data import *
from tools.kubernetes_util import run_kubectl_command
from tools.sop_flow_tool import match_sop, generate_and_run_sop_code, match_observation
import json
from llama_index.core import PromptTemplate
from agent.multiagent import JudgeRootCause

import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import argparse
# from traceloop.sdk import Traceloop


TYPE_INFO = False

BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "")
API_VERSION = os.environ.get("OPENAI_API_VERSION", "")
# os.environ["TRACELOOP_API_KEY"] = "tl_ce4f0d021f54443ca2125cfbfbbe4a90"
os.environ["TRACELOOP_API_KEY"] = "tl_13a59be395fd42acb9ea7723992a29c7"

if "deepseek" in MODEL_NAME.lower():
    llm = DeepSeek(
        api_key=API_KEY, api_base=BASE_URL, model=MODEL_NAME, temperature=0.5, timeout=300.0
    )
elif "gpt" in MODEL_NAME.lower():
    try:
        llm = AzureOpenAI(
            engine=MODEL_NAME,
            model=MODEL_NAME,
            azure_endpoint=BASE_URL,
            api_key=API_KEY,
            api_version=API_VERSION,
            timeout=300.0,
        )
    except:
        llm = OpenAI(
            model=MODEL_NAME,
            api_base=BASE_URL,
            api_key=API_KEY,
            api_version=API_VERSION,
            timeout=300.0,
        )
else:
    raise Exception("LLM model not founded.")


tool_list = [
    FunctionTool.from_defaults(fn)
    for fn in [
        # pod_analyze,
        # service_analyze,
        # statefulset_analyze,
        # node_analyze,
        # deployment_analyze,
        run_kubectl_command,
        time_series_anomaly_detection,
        trace_anomaly_detection,
        log_anomaly_detection,
        generate_and_run_sop_code,
        match_observation,
        match_sop,
        classifier,
        get_metric_values_offline,
        get_trace_values_offline,
        get_logs_offline,
        JudgeRootCause,
        query_system_information,
    ]
]


def update_react_prompt():
    react_system_header_str = """You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}

## Tool Usage Guidelines

1. **match_sop**: 
    - Use match_sop after classifier when fault type information is available. It helps find relevant SOP (Standard Operating Procedure) documentation.
2. **generate_and_run_sop_code**: 
    - After retrieving an SOP with match_sop, use generate_and_run_sop_code to generate and execute the SOP code.
    - When pod level and service level SOPs are matched, start with the service level first.
    - When multiple SOPs are matched, start with the higher score.
3. **match_observation**: 
    - Once the observation result is available after generate_and_run_sop_code, use match_observation to match the output to a likely fault type.
4. **classifier**: 
    - Use classifier at the beginning if you don't know the fault type.
5. **JudgeRootCause**:
    - Before give answer, use JudgeRootCause to analyze the fault information and determine the root causes.
6. Be FLEXIABLE and ADAPTIVE. 
    - For example, if code execution fails, regenerate the code and try again.

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought. In Thought, PLEASE first give several appropriate actions, analyze for each, and ultimately choose the most suitable one. (e.g. tool A .... tool B ...  The most suitable is ...)

MUST FOLLOW Tool Usage Guidelines mentioned above!

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

## Final Answer Format

Final answer SHOULD be a JSON of root causes (no more than three) of the failure (e.g. {{"anomaly type": "<anomaly type>", "root cause": [{{"location": "<instance_name>", "reason": "..."}}, {{"location": "<instance_name>", "reason": "..."}}, ...]}}).
If a pod is the root cause (e.g. adservice-0) , the corresponding service (e.g. adservice) might also be the root cause.

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""
    react_system_prompt = PromptTemplate(react_system_header_str)
    return react_system_prompt


async def main(fault):
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.llm = llm
    Settings.callback_manager = CallbackManager([token_counter])
    token_counter.reset_counts()
    path_length = 0
    if TYPE_INFO:
        fault_str = fault["Anomaly Description"]
    else:
        fault_str = fault["Anomaly Description"]
    agent = ReActAgent(tools=tool_list, max_iterations=10)
    ctx = Context(agent)
    prompt = (
        "A fault has occurred in the current Kubernetes system. "
        f"The fault description is: {fault_str}.\n\n"
        "Your task is to determine the anomaly type and root cause of this fault using the provided tools. "
        # "The root cause **must** be specific instance name (pod e.g. adservice-0, service e.g. adservice, node e.g. aiops-k8s-01) without any other information, and should be returned in the following JSON format (at least three, no more than five):\n\n"
        "The root cause **must** be specific instance name (pod e.g. adservice-0, service e.g. adservice, node e.g. aiops-k8s-01) without any other information, and should be returned in the following JSON format (no more than three):\n\n"
        "{\n"
        '  "anomaly type": "<anomaly type>",\n'
        '  "root cause": [\n'
        '    {"location": "<instance_name>", "reason": "<simple explanation>"},\n'
        '    {"location": "<instance_name>", "reason": "<simple explanation>"},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "🔧 **Tool Usage Guidelines — These steps are critical and must be followed carefully:**\n"
        "1. **match_sop**:\n"
        "   - Use match_sop when fault type information is available.\n"
        "   - It helps find relevant SOP (Standard Operating Procedure) documentation.\n"
        "2. **generate_and_run_sop_code**:\n"
        "   - After retrieving an SOP with match_sop, use generate_and_run_sop_code to generate and execute the corresponding SOP code.\n"
        "   - When pod level and service level SOPs are matched, start with the service level first."
        "   - When multiple SOPs are matched, start with the higher score."
        "3. **match_observation**:\n"
        "   - Once the observation is available after generate_and_run_sop_code, use match_observation to match the output to a likely fault type.\n\n"
        "4. **classifier**:\n"
        "   - Use classifier when fault type information is unavailable.\n\n"
        "5. **JudgeRootCause**:\n"
        "   - Before give final answer, use JudgeRootCause to analyze the fault information and determine the root causes.\n\n"
        "⚠️ **Important:**\n"
        "You must follow the above order unless exceptions occur. Be flexible and adaptive. For example:\n"
        "- If code execution fails, regenerate the code and try again.\n"
        "- If no matching SOP is found, analyze the fault and generate a new appropriate SOP.\n\n"
        "- If wrong metric name is used, use query_system_information to get the correct metric name.\n\n"
        # "During the Thought phase, PLEASE first give several appropriate actions, analyze for each, and ultimately choose the most suitable one.\n"
        "Think step by step, justify your actions, and always use the tools logically and effectively to pinpoint the root cause. If a pod is the root cause (e.g. adservice-0) , the corresponding service (e.g. adservice) might also be the root cause!"
    )
    agent.update_prompts({"react_header": update_react_prompt()})
    # prompt_dict = agent.get_prompts()
    # for k, v in prompt_dict.items():
    #     print(f"Prompt: {k}\n\nValue: {v.template}")

    reasoning_trace = []
    handler = agent.run(prompt)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            # print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
            print(f"\nObservation:\n{ev.tool_output}\n")
            reasoning_trace.append(
                {
                    "step": path_length,
                    "action": f"{ev.tool_name} Args: {ev.tool_kwargs}",
                    "observation": f"{ev.tool_output}",
                }
            )
            path_length += 1
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)

    response = await handler
    response = str(response).replace("```json", "").replace("```", "")
    response = json.loads(response)

    result = {
        "anomaly type": response["anomaly type"],
        "root cause": [item for item in response["root cause"]],
        "reasoning trace": reasoning_trace,
        "token count": token_counter.total_llm_token_count,
        "path length": path_length,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, default="server/playground/umodel/fault.json")
    parser.add_argument("-o", "--output_file", type=str, default="server/playground/umodel/answer.json")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        all_faults = json.load(f)

    results = []

    for fault in all_faults:
        # Traceloop.init()
        try:
            result = asyncio.run(main(fault))
        except:
            result = {
                "anomaly type": "",
                "root cause": [],
                "reasoning trace": [],
                "token count": 0,
                "path length": 0,
            }
        results.append(result)

        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)
