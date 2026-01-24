# # Create server parameters for stdio connection
# import os
# from pdb import run
# import sys
# from venv import create

# # Add src to sys.path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from typing import List, Optional, Literal
# import asyncio
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from mcp.client.sse import sse_client
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI

# from langchain_core.messages import SystemMessage, HumanMessage


# def get_llm():
#     return ChatOpenAI(
#         model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
#         api_key=os.environ.get("OPENAI_API_KEY"),
#         base_url=os.environ.get("OPENAI_BASE_URL"),  # 形如 http://host:port/v1
#         temperature=0,
#         # 强烈建议：限制输出，省 token，减少 429
#         max_tokens=512,  # 这个字段会映射成 max_completion_tokens
#         # 可选：减少重试风暴
#         max_retries=2,
#         timeout=60,
#     )

# async def run_mcp_agent_logic(
#     session: ClientSession, system_prompt: str, project_details: str, user_prompt: str
# ):
#     # Initialize the connection
#     await session.initialize()

#     llm = get_llm()

#     # Get tools
#     tools = await load_mcp_tools(session)


#     # Create and run the agent
#     agent = create_react_agent(llm, tools)

#     # msg = {
#     #     "messages": [
#     #         {
#     #             "role": "system",
#     #             "content": system_prompt + "\n\n" + project_details,
#     #         },
#     #         {"role": "user", "content": user_prompt},
#     #     ]
#     # }

#     msg = {
#         "messages": [
#             SystemMessage(content=system_prompt + "\n\n" + project_details),
#             HumanMessage(content=user_prompt),
#         ]
#     }
#     # agent_response = await agent.ainvoke(
#     #     # {"messages": "[('user', '列出所有的cms workspace')]" }
#     #     msg
#     # )

#     # return agent_response

#     events = agent.astream(msg, stream_mode="values")
#     final_response = ""
#     async for event in events:
#         try:
#             msgs = _extract_messages_from_event(event)
#             if not msgs:
#                 continue
#             last_msg = msgs[-1]
#             last_msg
#             print(
#                 f"\n[{last_msg.type.upper()}]:\n{'-'*10}\n{last_msg.content}",
#                 flush=True,
#             )
#             # for tool in (last_msg.tool_calls or []):
#             #     print(f"\n{tool}", flush=True)
#             if last_msg.type == "ai":
#                 final_response = last_msg.content

#         except Exception as e:
#             print(f"Error processing event: {e}", flush=True)

#     return final_response


# def _extract_messages_from_event(event: dict):
#     """从任意 astream event 里尽可能找到 messages 列表。"""
#     if not isinstance(event, dict):
#         return None

#     # 1) 直接就是 {"messages": [...]}
#     if "messages" in event and isinstance(event["messages"], list):
#         return event["messages"]

#     # 2) 常见：{"agent": {"messages": [...]}} / {"model": {"messages": [...]}} / {"tools": {"messages": [...]}}
#     for _, v in event.items():
#         if isinstance(v, dict) and "messages" in v and isinstance(v["messages"], list):
#             return v["messages"]

#     return None


# async def run_mcp_agent(
#     system_prompt: str,
#     project_details: str,
#     user_prompt: str,
#     connection_mode: Literal["sse", "stdio"] = "sse",
#     url: str = "http://127.0.0.1:8000/sse",
#     cmd: str = "python",
#     access_key_id: Optional[str] = None,
#     access_key_secret: Optional[str] = None,
#     sls_endpoints: Optional[str] = None,
#     cms_endpoints: Optional[str] = None,
# ):
#     try:
#         if connection_mode == "sse":
#             async with sse_client(url) as (read, write):
#                 async with ClientSession(read, write) as session:
#                     return await run_mcp_agent_logic(
#                         session, system_prompt, project_details, user_prompt
#                     )
#         elif connection_mode == "stdio":
#             cmd_args = ["-m", "mcp_server_aliyun_observability", "--transport", "stdio"]

#             # Add credentials if provided
#             if access_key_id and access_key_secret:
#                 cmd_args.extend(["--access-key-id", access_key_id])
#                 cmd_args.extend(["--access-key-secret", access_key_secret])

#             # Add endpoints if provided
#             if sls_endpoints:
#                 cmd_args.extend(["--sls-endpoints", sls_endpoints])
#             if cms_endpoints:
#                 cmd_args.extend(["--cms-endpoints", cms_endpoints])

#             server_params = StdioServerParameters(
#                 command=cmd,
#                 args=cmd_args,
#                 env=os.environ.copy(),  # important to pass credentials
#             )
#             async with stdio_client(server_params) as (read, write):
#                 async with ClientSession(read, write) as session:
#                     return await run_mcp_agent_logic(
#                         session, system_prompt, project_details, user_prompt
#                     )
#         else:
#             raise ValueError(f"Unsupported connection mode: {connection_mode}")
#     except* Exception as eg:
#         print("Sub-exceptions:")
#         for e in eg.exceptions:
#             print(" -", repr(e))
#         raise


# # Example usage
# asyncio.run(
#     run_mcp_agent(
#         system_prompt="",
# #         system_prompt="""
# # 你是SRE专家，使用阿里云可观测 2.0 (UModel) MCP 工具做故障定位。

# # 输入会给你 instance_type ∈ {pod,node,service}，它只表示分析入口对象类型。
# # 严禁猜测任何集合名/域名/指标名：包括 metric_domain_name、metric、log_set_name、event_set_name、trace_set_name 等。

# # 必须遵循流程：
# # 1) list_domains(workspace, regionId)
# # 2) umodel_search_entity_set(workspace, search_text=与instance_type相关关键词, regionId) 选择最匹配的 domain + entity_set_name
# # 3) umodel_list_data_set(workspace, domain, entity_set_name, regionId) 获取可用数据集合与参数选项
# # 4) 再调用 umodel_search_entities / umodel_get_entities 定位目标实体（若缺少实体标识则先给出可选实体列表并停止继续查询）
# # 5) 按需查询：优先 umodel_get_golden_metrics，其次 umodel_get_metrics(analysis_mode="anomaly_detection")，再结合 umodel_get_logs / umodel_get_events / umodel_search_traces 等
# # 6) 最终仅输出严格 JSON：{ "instance_type": "...", "root_cause": "...", "evidence": [...], "next_actions": [...] }

# # 当某类数据集不存在（例如 event_sets 为空），不要硬查该工具，直接跳过并在 evidence 说明缺失原因。""",
#         project_details="Your workspace is 'default-cms-1102382765107602-cn-heyuan' in region 'cn-heyuan', and the SLS project is 'default-cms-1102382765107602-cn-heyuan', logstore is **aiops-dataset-logs**. Use this information when configuring your data source connections.",
#         # project_details="",
#         user_prompt="时间范围start_time = 1766506202,end_time = 1766507462在service上发生了异常，对emailservice进行metric异常检测，分析是否service的指标出现了异常。返回json格式，包括异常指标名称和异常service的具体名称",
#     )
# )
