# Create server parameters for stdio connection
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-35-turbo-0125"),
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_BASE_URL", ""),
    temperature=0,
)


def build_project_details(workspace, region, project):
    # return
    return f"""Your UModel workspace is '{workspace}' in region '{region}', and the SLS project is '{project}', logstore is **aiops-dataset-logs**.
Use this information when configuring your data source connections.
"""


async def run_agent():
    async with sse_client("http://127.0.0.1:8000/sse") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            project_details = build_project_details(
                workspace="default-cms-1102382765107602-cn-heyuan",
                region="cn-heyuan",
                project="default-cms-1102382765107602-cn-heyuan",
            )
            agent_response = await agent.ainvoke(
                # {"messages": "[('user', '列出所有的cms workspace')]" }
                {
                    "messages": [
                        {"role": "system", "content": project_details},
                        {"role": "user", "content": "列出所有的cms workspace"},
                    ]
                }
            )
            return agent_response


# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
