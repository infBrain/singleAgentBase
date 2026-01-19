import os
import sys
import json
import datetime
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
# from langchain.agents import Tool # We will use @tool decorator instead

# Add src to sys.path to allow importing tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.multimodal_data import (
    time_series_anomaly_detection,
    trace_anomaly_detection,
    log_anomaly_detection,
    query_system_information,
    classifier
)

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
            temperature=0
        )
    elif "gpt" in MODEL_NAME.lower():
        if "azure" in BASE_URL.lower() or "api.cognitive.microsoft.com" in BASE_URL.lower():
             return AzureChatOpenAI(
                azure_endpoint=BASE_URL,
                openai_api_version=API_VERSION,
                azure_deployment=MODEL_NAME,
                openai_api_key=API_KEY,
                temperature=0
            ) 
        else:
             return ChatOpenAI(
                model=MODEL_NAME,
                openai_api_key=API_KEY,
                openai_api_base=BASE_URL,
                temperature=0
            )
    else:
        return ChatOpenAI(
            model=MODEL_NAME,
            openai_api_key=API_KEY,
            openai_api_base=BASE_URL,
            temperature=0
        )

# Define Tools
@tool
def search_metrics(query: str) -> str:
    """
    Search and analyze metric anomalies.
    Input should be a string with format: "start_time, end_time, metric_name, instance"
    Example: "2025-05-05T10:11:31Z, 2025-05-05T10:29:31Z, node_cpu_usage_rate, all"
    If you are not sure about metric_name or instance, use 'all'.
    """
    try:
        parts = [p.strip() for p in query.split(",")]
        if len(parts) != 4:
            return "Error: Input must have 4 parts separated by commas: start_time, end_time, metric_name, instance"
        start_time, end_time, metric_name, instance = parts
        return time_series_anomaly_detection(start_time, end_time, metric_name, instance)
    except Exception as e:
        return f"Error executing time_series_anomaly_detection: {str(e)}"

@tool
def search_traces(query: str) -> str:
    """
    Detect anomalies in traces.
    Input should be a string with format: "start_time, end_time"
    Example: "2025-05-05T10:11:31Z, 2025-05-05T10:29:31Z"
    """
    try:
        parts = [p.strip() for p in query.split(",")]
        if len(parts) != 2:
            return "Error: Input must have 2 parts separated by commas: start_time, end_time"
        start_time, end_time = parts
        return trace_anomaly_detection(start_time, end_time)
    except Exception as e:
        return f"Error executing trace_anomaly_detection: {str(e)}"

@tool
def search_logs(query: str) -> str:
    """
    Detect anomalies in logs.
    Input should be a string with format: "start_time, end_time"
    Example: "2025-05-05T10:11:31Z, 2025-05-05T10:29:31Z"
    """
    try:
        parts = [p.strip() for p in query.split(",")]
        if len(parts) != 2:
            return "Error: Input must have 2 parts separated by commas: start_time, end_time"
        start_time, end_time = parts
        return log_anomaly_detection(start_time, end_time)
    except Exception as e:
        return f"Error executing log_anomaly_detection: {str(e)}"

@tool
def get_system_info(query: str) -> str:
    """
    Retrieve system information.
    Input is a natural language question about the system.
    """
    return query_system_information(query)

@tool
def analyze_fault_type(query: str) -> str:
    """
    Analyze anomaly detection results to determine the main fault type.
    This tool uses a classifier to identify the likely failure category (e.g., Pod Memory, Network Delay) based on metric patterns.
    Input should be a string with format: "start_time, end_time"
    Example: "2025-05-05T10:11:31Z, 2025-05-05T10:29:31Z"
    """
    try:
        parts = [p.strip() for p in query.split(",")]
        if len(parts) != 2:
            return "Error: Input must have 2 parts separated by commas: start_time, end_time"
        start_time, end_time = parts
        return classifier(start_time, end_time)
    except Exception as e:
        return f"Error executing classifier: {str(e)}"

tools = [analyze_fault_type, search_metrics, search_traces, search_logs, get_system_info]

def run_rca_agent(start_time: str, end_time: str):
    llm = get_llm()
    # Create the agent
    system_prompt = f"""You are a Site Reliability Engineer (SRE) agent responsible for Root Cause Analysis (RCA).
Your task is to determine the anomaly type and root cause of the fault that occurred between {start_time} and {end_time}.

The root cause **must** be specific instance name (pod e.g. adservice-0, service e.g. adservice, node e.g. aiops-k8s-01) without any other information, and should be returned in the following JSON format (no more than three):

{{
  "anomaly type": "<anomaly type>",
  "root cause": [
    {{"location": "<instance_name>", "reason": "<simple explanation>"}},
    {{"location": "<instance_name>", "reason": "<simple explanation>"}},
    ...
  ]
}}

🔧 **Tool Usage Guidelines — These steps are critical and must be followed carefully:**
1. **analyze_fault_type**:
   - Use this tool FIRST to get a high-level classification of the fault (e.g., "Pod CPU", "Network Delay").
   - This guides your subsequent investigation.
2. **search_metrics**:
   - Verify the classification and identify specific affected components.
   - You can start with broad metrics ('all') or drill down based on the classification.
3. **search_traces**:
   - Use traces to identify latency issues or call chain errors.
   - This can specifically help pinpoint service-level issues.
4. **search_logs**:
   - Check logs for the specific time period to find error messages or exceptions.
5. **get_system_info**:
   - Use this to understand the system topology or unrelated configuration information if needed.

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
Just output the raw JSON string.
"""

    agent_executor = create_react_agent(llm, tools, prompt=system_prompt)

    try:
        response = agent_executor.invoke({"messages": [("user", f"Analyze the system status from {start_time} to {end_time}")]})
        result_content = response["messages"][-1].content
        
        print("Agent raw output:")
        print(result_content)

        # Try to parse the result as JSON to ensure it matches the format
        cleaned_result = result_content.strip()
        
        # If the model still ignores instructions and adds "Thought:", try to split
        if "Answer:" in cleaned_result:
             cleaned_result = cleaned_result.split("Answer:")[-1].strip()
        
        # Strip markdown codes if present
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
        elif cleaned_result.startswith("```"):
            cleaned_result = cleaned_result[3:]
        
        if cleaned_result.endswith("```"):
            cleaned_result = cleaned_result[:-3]
        
        cleaned_result = cleaned_result.strip()

        try:
             json_result = json.loads(cleaned_result)
        except json.JSONDecodeError:
             # Find the first '{' and last '}'
             start_idx = cleaned_result.find("{")
             end_idx = cleaned_result.rfind("}")
             if start_idx != -1 and end_idx != -1:
                 maybe_json = cleaned_result[start_idx:end_idx+1]
                 try:
                     json_result = json.loads(maybe_json)
                 except:
                     json_result = {
                         "root_cause": "unknown",
                         "raw_output": result_content,
                         "error": "Failed to parse LLM output as specific JSON format"
                     }
             else:
                 json_result = {
                     "root_cause": "unknown",
                     "raw_output": result_content,
                     "error": "Failed to parse LLM output as specific JSON format"
                 }

    except Exception as e:
        json_result = {
            "root_cause": "unknown",
            "error": f"Agent failed to run: {str(e)}"
        }

    json_result["start_time"] = start_time
    json_result["end_time"] = end_time

    # Ensure result directory exists
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rca_result_{timestamp}.json"
    filepath = os.path.join(result_dir, filename)
    
    # with open(filepath, "w") as f:
    #     json.dump(json_result, f, indent=4)
    
    # print(f"RCA result saved to {filepath}")
    return json_result

if __name__ == "__main__":
    # Example usage for testing
    # You can accept args from command line if needed
    if len(sys.argv) >= 3:
        st = sys.argv[1]
        et = sys.argv[2]
        run_rca_agent(st, et)
    else:
        # Default test case from multimodal_data.py
        # result = time_series_anomaly_detection("2025-06-05T16:10:02Z", "2025-06-05T16:31:02Z")
        print("Usage: python rca_agent.py <start_time> <end_time>")
        print("Using default test time from multimodal_data.py example")
        run_rca_agent("2025-06-05T16:10:02Z", "2025-06-05T16:31:02Z")
