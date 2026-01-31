import os
from re import L
import sys
import json
import datetime
from typing import List, Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pyparsing import Opt

# from langchain.agents import Tool # We will use @tool decorator instead

# Add src to sys.path to allow importing tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup LLM based on environment variables (similar to tools/llm_chat.py)
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-35-turbo-0125")
API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-03-01-preview")


def get_llm():
    if "deepseek" in MODEL_NAME.lower():
        return ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0,
        )
    elif "gpt" in MODEL_NAME.lower():
        if (
            "azure" in BASE_URL.lower()
            or "api.cognitive.microsoft.com" in BASE_URL.lower()
        ):
            return AzureChatOpenAI(
                azure_endpoint=BASE_URL,
                openai_api_version=API_VERSION,
                azure_deployment=MODEL_NAME,
                openai_api_key=API_KEY,
                temperature=0,
            )
        else:
            return ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=API_KEY,
                openai_api_base=BASE_URL,
                temperature=0,
            )
    else:
        return ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0,
        )


def run_traditional_agent(system_prompt: str, user_prompt: str, tools: Optional[List] = None):
    llm = get_llm()

    agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

    try:
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
        print(f"Traditional Agent failed to run: {str(e)}")
        cleaned_result = f"{{Traditional Agent failed to run: {str(e)}}}"

    return cleaned_result


# if __name__ == "__main__":
#     # Example usage for testing
#     # You can accept args from command line if needed
#     if len(sys.argv) >= 3:
#         st = sys.argv[1]
#         et = sys.argv[2]
#         run_rca_agent(st, et)
#     else:
#         # Default test case from multimodal_data.py
#         # result = time_series_anomaly_detection("2025-06-05T16:10:02Z", "2025-06-05T16:31:02Z")
#         print("Usage: python rca_agent.py <start_time> <end_time>")
#         print("Using default test time from multimodal_data.py example")
#         run_rca_agent("2025-06-05T16:10:02Z", "2025-06-05T16:31:02Z")
