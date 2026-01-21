import os
import sys
import asyncio
import argparse
from typing import List, Optional, Literal

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
# from langchain.agents import AgentExecutor

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, ListToolsResult

# Configuration
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-35-turbo-0125")
API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-03-01-preview")

def get_llm():
    """Configures and returns the LLM instance based on environment variables."""
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

async def mcp_tool_wrapper(name: str, arguments: dict, session: ClientSession) -> str:
    """Generic wrapper to call MCP tools."""
    try:
        result: CallToolResult = await session.call_tool(name, arguments)
        
        # Format the output
        output_text = ""
        if result.content:
            for content in result.content:
                if content.type == "text":
                    output_text += content.text + "\n"
                elif content.type == "resource": # Handle resources if returned
                    output_text += f"[Resource: {content.uri}]\n"
        
        if result.isError:
            return f"Error executing tool {name}: {output_text}"
            
        return output_text.strip()
    except Exception as e:
        return f"Exception calling tool {name}: {str(e)}"

async def convert_mcp_tools_to_langchain(session: ClientSession) -> List[StructuredTool]:
    """Fetches tools from MCP server and converts them to LangChain StructuredTools."""
    
    # 1. List tools from MCP
    response: ListToolsResult = await session.list_tools()
    langchain_tools = []

    print(f"Found {len(response.tools)} tools from MCP Server.")

    for tool_schema in response.tools:
        # Capture current values for closure
        current_name = tool_schema.name
        current_description = tool_schema.description or f"Tool {current_name}"
        current_input_schema = tool_schema.inputSchema

        # Create an async function for this specific tool
        # We need to bind the session and tool name
        async def _run_tool(**kwargs):
            return await mcp_tool_wrapper(current_name, kwargs, session)
        
        # Create LangChain tool
        lc_tool = StructuredTool.from_function(
            func=None, # Synchronous version
            coroutine=_run_tool, # Async version
            name=current_name,
            description=current_description,
        )
        
        langchain_tools.append(lc_tool)

    return langchain_tools

async def run_mcp_agent_logic(session: ClientSession, query: str):
    """The common agent logic once session is established."""
    # Initialize
    await session.initialize()
    
    # Get Tools
    tools = await convert_mcp_tools_to_langchain(session)
    
    if not tools:
        print("No tools found via MCP. Exiting.")
        return "No tools found via MCP."

    # Setup Agent
    llm = get_llm()
    print('=================\n',tools[:-1])
    agent_executor = create_react_agent(llm, tools[:-1])

    # print(f"\n--- Starting Root Cause Analysis Task ---\nQuery: {query}\n")

    # Execute
    events = agent_executor.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values"
    )

    final_response = ""
    async for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]
            print(f"\n[{last_msg.type.upper()}]:\n{last_msg.content}")
            if last_msg.type == "ai":
                final_response = last_msg.content
                
    return final_response

async def run_mcp_agent(query: str, 
                   connection_mode: Literal["sse", "stdio"] = "stdio", 
                   url_or_cmd: str = "python",
                   access_key_id: Optional[str] = None,
                   access_key_secret: Optional[str] = None,
                   sls_endpoints: Optional[str] = None,
                   cms_endpoints: Optional[str] = None):
    """
    Main execution entry point.
    
    :param query: The query to run info.
    :param connection_mode: "sse" or "stdio"
    :param url_or_cmd: If sse, the URL. If stdio, the command (e.g., 'python').
    :param access_key_id: Optional Alibaba Cloud AccessKey ID
    :param access_key_secret: Optional Alibaba Cloud AccessKey Secret
    :param sls_endpoints: Optional specific SLS endpoints (e.g., "cn-heyuan=...")
    :param cms_endpoints: Optional specific CMS endpoints
    :return: The final response from the agent.
    """
    
    if connection_mode == "sse":
        print(f"Connecting to MCP Server via SSE at {url_or_cmd}...")
        async with sse_client(url_or_cmd) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                return await run_mcp_agent_logic(session, query)
                
    elif connection_mode == "stdio":
        cmd_args = ["-m", "mcp_server_aliyun_observability", "--transport", "stdio"]
        
        # Add credentials if provided
        if access_key_id and access_key_secret:
            cmd_args.extend(["--access-key-id", access_key_id])
            cmd_args.extend(["--access-key-secret", access_key_secret])

        # Add endpoints if provided
        if sls_endpoints:
            cmd_args.extend(["--sls-endpoints", sls_endpoints])
        if cms_endpoints:
            cmd_args.extend(["--cms-endpoints", cms_endpoints])
            
        print(f"Starting MCP Server via Stdio: {url_or_cmd} {' '.join(cmd_args)}")
        
        # We assume url_or_cmd is the python executable properly configured
        server_params = StdioServerParameters(
            command=url_or_cmd,
            args=cmd_args,
            env=os.environ.copy() # important to pass credentials
        )
        
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                return await run_mcp_agent_logic(session, query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RCA Agent with MCP Tools")
    parser.add_argument("--query", type=str, help="The root cause analysis query", default="请分析系统最近是否有异常？")
    parser.add_argument("--mode", type=str, choices=["sse", "stdio"], default="stdio", help="Connection mode")
    parser.add_argument("--target", type=str, default="python", help="SSE URL or Python Executable Path")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_mcp_agent(args.query, args.mode, args.target))
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
