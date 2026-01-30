import os
import sys
import asyncio
from typing import List, Optional, Literal

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.mcp_tools import (
    guide_intro,
    set_mcp_session,
    clear_mcp_session,
    introduction,
    list_workspace,
    list_domains,
    umodel_search_entity_set,
    umodel_list_data_set,
    umodel_list_related_entity_set,
    umodel_get_entities,
    umodel_search_entities,
    umodel_get_neighbor_entities,
    umodel_get_golden_metrics,
    umodel_get_metrics,
    umodel_get_relation_metrics,
    umodel_get_logs,
    umodel_get_events,
    umodel_search_traces,
    umodel_get_traces,
    umodel_get_profiles,
    sls_text_to_sql,
    sls_execute_sql,
    sls_get_context_logs,
    sls_log_explore,
    sls_log_compare,
    sls_list_projects,
    sls_execute_spl,
    sls_list_logstores,
    cms_text_to_promql,
    cms_execute_promql,
)


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


def get_mcp_tools() -> List:
    return [
        guide_intro,
        introduction,
        list_workspace,
        list_domains,
        umodel_search_entity_set,
        umodel_list_data_set,
        umodel_list_related_entity_set,
        umodel_get_entities,
        umodel_search_entities,
        umodel_get_neighbor_entities,
        umodel_get_golden_metrics,
        umodel_get_metrics,
        umodel_get_relation_metrics,
        umodel_get_logs,
        umodel_get_events,
        umodel_search_traces,
        umodel_get_traces,
        umodel_get_profiles,
        sls_text_to_sql,
        sls_execute_sql,
        sls_get_context_logs,
        sls_log_explore,
        sls_log_compare,
        sls_list_projects,
        sls_execute_spl,
        sls_list_logstores,
        cms_text_to_promql,
        cms_execute_promql,
    ]


async def run_mcp_agent_logic(
    session: ClientSession, system_prompt: str, project_details: str, user_prompt: str
) -> str:
    await session.initialize()
    set_mcp_session(session)

    llm = get_llm()
    tools = get_mcp_tools()

    # use call to load tools dynamically instead of loading all at once
    # tools = await load_mcp_tools(session) 
    # tools.append(guide_intro)

    agent = create_react_agent(llm, tools)

    msg = {
        "messages": [
            SystemMessage(content=system_prompt + "\n\n" + project_details),
            HumanMessage(content=user_prompt),
        ]
    }

    events = agent.astream(msg, stream_mode="values")
    final_response = ""
    async for event in events:
        try:
            msgs = _extract_messages_from_event(event)
            if not msgs:
                continue
            last_msg = msgs[-1]
            print(
                f"\n[{last_msg.type.upper()}]:\n{'-'*10}\n{last_msg.content}",
                flush=True,
            )
            if last_msg.type == "ai":
                final_response = last_msg.content
        except Exception as e:
            print(f"Error processing event: {e}", flush=True)

    clear_mcp_session()
    return final_response


def _extract_messages_from_event(event: dict):
    if not isinstance(event, dict):
        return None

    if "messages" in event and isinstance(event["messages"], list):
        return event["messages"]

    for _, value in event.items():
        if (
            isinstance(value, dict)
            and "messages" in value
            and isinstance(value["messages"], list)
        ):
            return value["messages"]

    return None


async def run_mcp_agent(
    system_prompt: str,
    project_details: str,
    user_prompt: str,
    connection_mode: Literal["sse", "stdio"] = "sse",
    url: str = "http://127.0.0.1:8000/sse",
    cmd: str = "python",
    access_key_id: Optional[str] = None,
    access_key_secret: Optional[str] = None,
    sls_endpoints: Optional[str] = None,
    cms_endpoints: Optional[str] = None,
) -> str:
    try:
        if connection_mode == "sse":
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    return await run_mcp_agent_logic(
                        session, system_prompt, project_details, user_prompt
                    )

        if connection_mode == "stdio":
            cmd_args = ["-m", "mcp_server_aliyun_observability", "--transport", "stdio"]

            if access_key_id and access_key_secret:
                cmd_args.extend(["--access-key-id", access_key_id])
                cmd_args.extend(["--access-key-secret", access_key_secret])

            if sls_endpoints:
                cmd_args.extend(["--sls-endpoints", sls_endpoints])
            if cms_endpoints:
                cmd_args.extend(["--cms-endpoints", cms_endpoints])

            server_params = StdioServerParameters(
                command=cmd,
                args=cmd_args,
                env=os.environ.copy(),
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    return await run_mcp_agent_logic(
                        session, system_prompt, project_details, user_prompt
                    )

        raise ValueError(f"Unsupported connection mode: {connection_mode}")
    except* Exception as eg:
        print("Sub-exceptions:")
        for exc in eg.exceptions:
            print(" -", repr(exc))
        # raise


# if __name__ == "__main__":
#     asyncio.run(
#         run_mcp_agent(
#             system_prompt="你是一个sre运维专家，给出异常时间段后，请定位到异常根因。如果是service异常，那么相关pod也会出现异常，需要对相关pod进行查询分析，且相关pod的名字为service-0、service-1这种格式。只需要返回json格式的结果即可，根因定位的粒度最终定位到指定粒度上。",
#             # project_details="区域为cn-heyuan",
#             # user_prompt="列出所有可用的工作空间",
#             project_details="Your workspace is 'zy-aiops-challenges-2025' in region 'cn-heyuan', and the SLS project is 'default-cms-1102382765107602-cn-heyuan', logstore is **aiops-dataset-logs**. Use this information when configuring your data source connections.",
#             # user_prompt="时间范围start_time = 1766506202,end_time = 1766507462在service上发生了异常，分析具体是哪个service的指标出现了异常，并定位到故障根因。最终返回json格式，包括异常指标名称和异常service的具体名称 \n 如果无法定位到具体的service异常，就将全部的service和相关pod列出来，对其每个指标进行分析，必须得到具体的异常指标和并定位到根因异常service名称，如果全部查询后仍无法定位到具体的异常service，则返回无法定位。分析具体是哪个service的指标出现了异常，并定位到故障根因。最终返回json格式，包括异常指标名称和异常service的具体名称 \n 如果无法定位到具体的service异常，就将全部的service和相关pod列出来，对其每个指标进行分析，必须得到具体的异常指标和并定位到根因异常service名称，如果全部查询后仍无法定位到具体的异常service，则返回无法定位。",
#             user_prompt="时间范围start_time = 1766506202,end_time = 1766507462在service上发生了异常，查出emailservice和相关pod的指标，返回有异常的数据，返回结果为json格式",
#         )
#     )
