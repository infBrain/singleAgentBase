# -*- coding: utf-8 -*-
import asyncio
import os
import sys
import json

import argparse

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_comparison import run_mcp_only

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RCA Task with MCP")
    parser.add_argument("--workspace", default="aiops-challenges-2025", help="UModel Workspace Name")
    parser.add_argument("--region", default="cn-heyuan", help="Region ID")
    parser.add_argument("--project", default="cms-aiops-challenges-2025-dbj7stfif6apnzrb", help="SLS Project Name")
    parser.add_argument("--start-time", default="2025-06-05T16:10:02Z", help="Start time in ISO format")
    parser.add_argument("--end-time", default="2025-06-05T16:31:02Z", help="End time in ISO format")
    parser.add_argument("--task", default="最近系统出现了响应时间变慢的情况。请你检查一下最近1小时内的 metrics，看看有没有异常。", help="Description of the problem")
    parser.add_argument("--sls-endpoints", default="cn-heyuan=cms-aiops-challenges-2025-dbj7stfif6apnzrb", help="Override SLS endpoints (e.g. 'cn-region=host')")
    parser.add_argument("--cms-endpoints", default="cn-heyuan=aiops-challenges-2025", help="Override CMS endpoints (e.g. 'cn-region=host')")
    
    args = parser.parse_args()

    try:
        result = asyncio.run(run_mcp_only(
            workspace=args.workspace,
            region=args.region,
            project=args.project,
            start_time=args.start_time,
            end_time=args.end_time,
            task_description=args.task,
            sls_endpoints=args.sls_endpoints,
            cms_endpoints=args.cms_endpoints
        ))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"An error occurred: {e}")
