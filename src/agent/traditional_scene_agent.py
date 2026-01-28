import os
import re
import sys
import json
import datetime
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

# from langchain.agents import Tool # We will use @tool decorator instead

# Add src to sys.path to allow importing tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.traditional_tools import (
    # analyze_fault_type,
    detect_metrics,
    detect_traces,
    detect_logs,
    get_logs,
    get_metrics,
    get_traces,
    get_system_info,
)

# Setup LLM based on environment variables (similar to tools/llm_chat.py)
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-35-turbo-0125")
API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-03-01-preview")


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0,
        max_tokens=512,
        max_retries=30,
        timeout=60,
    )

tools = [
    # analyze_fault_type,
    detect_metrics,
    detect_traces,
    detect_logs,
    get_logs,
    get_metrics,
    get_traces,
    get_system_info,
]


def run_rca_agent(start_time: str, end_time: str, system_prompt: str, user_prompt: str):
    llm = get_llm()

    agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

    try:
        # response = agent_executor.invoke({"messages": [("user", f"Analyze the system status from {start_time} to {end_time}")]})
        response = agent_executor.invoke({"messages": [("user", user_prompt)]})
        result_content = response["messages"][-1].content

        print("Agent raw output:")
        print(result_content)

        # Try to parse the result as JSON to ensure it matches the format
        cleaned_result = result_content.strip()

        # If the model still ignores instructions and adds "Thought:", try to split
        if "Answer:" in cleaned_result:
            cleaned_result = cleaned_result.split("Answer:")[-1].strip()

    #     # Strip markdown codes if present
    #     if cleaned_result.startswith("```json"):
    #         cleaned_result = cleaned_result[7:]
    #     elif cleaned_result.startswith("```"):
    #         cleaned_result = cleaned_result[3:]

    #     if cleaned_result.endswith("```"):
    #         cleaned_result = cleaned_result[:-3]

    #     cleaned_result = cleaned_result.strip()

    #     try:
    #         json_result = json.loads(cleaned_result)
    #     except json.JSONDecodeError:
    #         # Find the first '{' and last '}'
    #         start_idx = cleaned_result.find("{")
    #         end_idx = cleaned_result.rfind("}")
    #         if start_idx != -1 and end_idx != -1:
    #             maybe_json = cleaned_result[start_idx : end_idx + 1]
    #             try:
    #                 json_result = json.loads(maybe_json)
    #             except:
    #                 json_result = {
    #                     "root_cause": "unknown",
    #                     "raw_output": result_content,
    #                     "error": "Failed to parse LLM output as specific JSON format",
    #                 }
    #         else:
    #             json_result = {
    #                 "root_cause": "unknown",
    #                 "raw_output": result_content,
    #                 "error": "Failed to parse LLM output as specific JSON format",
    #             }

    # except Exception as e:
    #     json_result = {
    #         "root_cause": "unknown",
    #         "error": f"Agent failed to run: {str(e)}",
    #     }

    # return json_result

    except Exception as e:
        cleaned_result = f"Agent failed to run: {str(e)}"

    return cleaned_result
