


import os
from pdb import run
import sys
from tracemalloc import start
from turtle import end_fill
from agent.mcp_rca_agent import run_mcp_agent
from tools.rca_output import parse_rca_json_output
from tools.utils import _convert_to_beijing


def build_system_prompt(start_time, end_time):
    return f"""列出所有的workspace"""


def build_user_message(start_time, end_time):
    return f"列出所有的workspace,查询{start_time} to {end_time}的日志数据"


def build_project_details(workspace, region, project):
    return f"""Your UModel workspace is '{workspace}' in region '{region}', and the SLS project is '{project}'.
Use this information when configuring your data source connections.
"""

async def run_mcp_only(
    start_time,
    end_time,
    sls_endpoints="cn-heyuan=cn-heyuan.log.aliyuncs.com",
    cms_endpoints="cn-heyuan=cms.cn-heyuan.aliyuncs.com",
    ground_truth=None,
    uuid=None,
    delay=201*24*60,
):
    prompt_start_time=_convert_to_beijing(start_time, delay)
    prompt_end_time=_convert_to_beijing(end_time, delay)
    system_prompt = build_system_prompt(prompt_start_time, prompt_end_time)
    user_message = build_user_message(prompt_start_time, prompt_end_time)
    project_details = build_project_details(
        workspace="default-cms-1102382765107602-cn-heyuan",
        region="cn-heyuan",
        project="default-cms-1102382765107602-cn-heyuan",
    )
    mcp_query = f"{system_prompt}\n{project_details}\nUser Request:\n{user_message}\n"

    python_executable = sys.executable
    access_key_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID")
    access_key_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    mcp_result_text = await run_mcp_agent(
        query=mcp_query,
        connection_mode="stdio",
        url_or_cmd=python_executable,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        # sls_endpoints=sls_endpoints if sls_endpoints else "cn-heyuan=default-cms-1102382765107602-cn-heyuan",
        # cms_endpoints=cms_endpoints if cms_endpoints else "cn-heyuan=default-cms-1102382765107602-cn-heyuan",
        sls_endpoints=sls_endpoints,
        cms_endpoints=cms_endpoints
    )

    return mcp_result_text




import asyncio
if __name__ == "__main__":
    start_time = "2025-06-05T16:10:02Z"
    end_time = "2025-06-05T16:31:02Z"

    result = asyncio.run(
        run_mcp_only(
            start_time=start_time,
            end_time=end_time,
            sls_endpoints="cn-heyuan=cn-heyuan.log.aliyuncs.com",
            cms_endpoints="cn-heyuan=cms.cn-heyuan.aliyuncs.com",
        ))
    
    print(result)