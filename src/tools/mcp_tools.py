import datetime
import time
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from mcp import ClientSession
from mcp.types import CallToolResult
from pandas import Timestamp

_MCP_SESSION: Optional[ClientSession] = None


def set_mcp_session(session: ClientSession) -> None:
    """设置全局 MCP 会话，供 @tool 封装使用。"""
    global _MCP_SESSION
    _MCP_SESSION = session


def clear_mcp_session() -> None:
    """清理全局 MCP 会话。"""
    global _MCP_SESSION
    _MCP_SESSION = None


def _require_session() -> ClientSession:
    if _MCP_SESSION is None:
        raise RuntimeError(
            "MCP session is not set. Call set_mcp_session(session) first."
        )
    return _MCP_SESSION


def _clean_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in arguments.items() if value is not None}


async def _call_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Generic MCP call wrapper for query-style tools."""
    session = _require_session()
    print(f"Calling MCP tool: {name} with arguments: {arguments}")
    result: CallToolResult = await session.call_tool(name, arguments)
    output_text = ""

    if result.content:
        for content in result.content:
            if content.type == "text":
                output_text += content.text + "\n"
            elif content.type == "resource":
                output_text += f"[Resource: {content.uri}]\n"

    if result.isError:
        return f"Error executing tool {name}: {output_text.strip()}"

    return output_text.strip()


@tool
async def introduction() -> str:
    """获取阿里云可观测性MCP Server的介绍和使用说明。

    功能概述：返回阿里云可观测性 MCP Server 的服务概述、核心能力和使用限制说明。
    使用场景：首次接入时了解服务能力和使用前提。
    注意事项：此工具不需要任何参数，可直接调用；返回信息包含各层工具的使用前提条件。
    """
    return await _call_tool("introduction", {})


@tool
async def guide_intro() -> str:
    """Return guidance for MCP tool usage flow and norms.

    Overview: Provides expectations, call order, parameter choices, and error handling tips.
    Use case: Obtain a unified MCP workflow guide before executing tasks.
    Notes: Returns guidance text only and does not call MCP APIs.
    """
    return (
        "MCP Tool Usage Guide:\n"
        "1) Identify regionId/workspace/time range first; call list_workspace if missing.\n"
        "2) For entity info, start with umodel_search_entity_set/umodel_get_entities.\n"
        "3) Before metrics/logs/events/traces, use umodel_list_data_set to confirm set names.\n"
        "4) Query tools return text only; double-check field names and time windows if needed.\n"
        "5) On empty results or errors, narrow the window and verify domain/entity_set_name.\n"
        "6) Keep each call focused; avoid oversized queries and irrelevant parameters."
    )


@tool
async def list_workspace(regionId: str) -> str:
    """列出可用的CMS工作空间。

    功能概述：获取指定区域内可用的Cloud Monitor Service (CMS)工作空间列表。
    参数说明：regionId（阿里云区域标识符，如"cn-hangzhou"）。
    返回结果：包含工作空间信息的字典（workspaces/total_count/region）。
    """
    return await _call_tool("list_workspace", {"regionId": regionId})


@tool
async def list_domains(workspace: str, regionId: str) -> str:
    """列出所有可用的实体域。

    功能概述：获取系统中所有可用的实体域（domain）列表。
    使用场景：了解支持的实体域与选择正确domain参数。
    返回数据：每个域包含 __domain__（域名称）与 cnt（该域实体总数）。
    """
    return await _call_tool(
        "list_domains", {"workspace": workspace, "regionId": regionId}
    )


@tool
async def umodel_search_entity_set(
    workspace: str,
    search_text: str,
    regionId: str,
    domain: Optional[str] = None,
    entity_set_name: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """搜索实体集合，支持全文搜索并按相关度排序。

    功能概述：在UModel元数据中搜索实体集合定义，返回domain/name/display_name等元数据信息。
    使用场景：实体集合发现与元数据探索。
    参数：search_text、workspace、regionId（必需）；domain、entity_set_name、limit（可选）。
    """
    return await _call_tool(
        "umodel_search_entity_set",
        _clean_args(
            {
                "workspace": workspace,
                "search_text": search_text,
                "regionId": regionId,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "limit": limit,
            }
        ),
    )


@tool
async def umodel_list_data_set(
    workspace: str,
    domain: str,
    entity_set_name: str,
    regionId: str,
    data_set_types: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """列出指定实体的可用数据集合，为其他PaaS工具提供参数选项。

    功能概述：获取指定实体域与类型下可用的数据集合信息。
    参数说明：data_set_types支持metric_set/log_set/event_set/trace_set/entity_set_name。
    工具依赖：为umodel_get_metrics/logs/events/traces提供可选参数。
    示例：
        umodel_list_data_set(
            workspace="default-cms-xxx",
            domain="aiops",
            entity_set_name="aiops.service",
            regionId="cn-heyuan",
            data_set_types="metric_set"
        )
    """
    return await _call_tool(
        "umodel_list_data_set",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "regionId": regionId,
                "data_set_types": data_set_types,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_list_related_entity_set(
    workspace: str,
    domain: str,
    entity_set_name: str,
    regionId: str,
    relation_type: Optional[str] = None,
    direction: Optional[str] = None,
    detail: Optional[bool] = None,
) -> str:
    """列出与指定实体集合相关的其他实体集合。

    功能概述：发现与源实体集合存在关系定义的其他实体集合类型。
    功能特点：支持方向控制与关系类型过滤；元数据级别展示关系定义。
    参数：workspace、domain、entity_set_name、regionId（必需）；relation_type、direction、detail（可选）。
    """
    return await _call_tool(
        "umodel_list_related_entity_set",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "regionId": regionId,
                "relation_type": relation_type,
                "direction": direction,
                "detail": detail,
            }
        ),
    )


@tool
async def umodel_get_entities(
    workspace: str,
    domain: str,
    entity_set_name: str,
    regionId: str,
    entity_ids: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """获取实体信息的PaaS API工具。

    功能概述：检索实体基础信息，支持分页查询与精确ID查询。
    工具分工：本工具用于基础实体信息；模糊搜索请用umodel_search_entities。
    参数：workspace、domain、entity_set_name、regionId（必需）；entity_ids、from_time、to_time、limit（可选）。
    示例：
        # 获取实体并使用返回的 __entity_id__ 进行后续查询
        umodel_get_entities(
            workspace="default-cms-xxx",
            domain="aiops",
            entity_set_name="aiops.service",
            regionId="cn-heyuan",
            limit=20
        )
    """
    return await _call_tool(
        "umodel_get_entities",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "regionId": regionId,
                "entity_ids": entity_ids,
                "from_time": from_time,
                "to_time": to_time,
                # "from_time": int(datetime.datetime.now().timestamp() - 86400 * 30),
                # "to_time": int(datetime.datetime.now().timestamp()),
                "limit": limit,
            }
        ),
    )


@tool
async def umodel_search_entities(
    workspace: str,
    domain: str,
    entity_set_name: str,
    search_text: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """基于关键词搜索实体信息。

    功能概述：在指定实体集合中按关键词进行模糊搜索与全文检索。
    功能特点：支持数量控制与灵活过滤。
    参数：workspace、domain、entity_set_name、search_text、regionId（必需）；from_time、to_time、limit（可选）。
    """
    return await _call_tool(
        "umodel_search_entities",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "search_text": search_text,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
                # "from_time": int(datetime.datetime.now().timestamp() - 86400 * 30),
                # "to_time": int(datetime.datetime.now().timestamp()),
                "limit": limit,
            }
        ),
    )


@tool
async def umodel_get_neighbor_entities(
    workspace: str,
    domain: str,
    entity_set_name: str,
    entity_id: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """获取指定实体的邻居实体信息。

    功能概述：基于实体间关联关系查找邻居实体。
    使用场景：依赖分析、关联发现、故障影响评估、拓扑构建。
    参数：workspace、domain、entity_set_name、entity_id、regionId（必需）；from_time、to_time、limit（可选）。
    示例：
        # entity_id 使用 umodel_get_entities 返回的 __entity_id__
        umodel_get_neighbor_entities(
            workspace="default-cms-xxx",
            domain="aiops",
            entity_set_name="aiops.service",
            entity_id="<__entity_id__>",
            regionId="cn-heyuan"
        )
    """
    return await _call_tool(
        "umodel_get_neighbor_entities",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "entity_id": entity_id,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
                # "from_time": int(datetime.datetime.now().timestamp() - 86400 * 30),
                # "to_time": int(datetime.datetime.now().timestamp()),
                "limit": limit,
            }
        ),
    )


@tool
async def umodel_get_golden_metrics(
    workspace: str,
    domain: str,
    entity_set_name: str,
    regionId: str,
    entity_ids: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取实体的黄金指标（关键性能指标）数据。

    参数获取：domain/entity_set_name通过umodel_search_entity_set，entity_ids可通过umodel_get_entities。
    参数：workspace、domain、entity_set_name、regionId（必需）；entity_ids、from_time、to_time（可选）。
    示例：
        umodel_get_golden_metrics(
            workspace="default-cms-xxx",
            domain="aiops",
            entity_set_name="aiops.service",
            regionId="cn-heyuan",
            from_time=1766506202,
            to_time=1766507462
        )
    """
    return await _call_tool(
        "umodel_get_golden_metrics",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "regionId": regionId,
                "entity_ids": entity_ids,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_get_metrics(
    workspace: str,
    domain: str,
    entity_set_name: str,
    metric_domain_name: str,
    metric: str,
    regionId: str,
    entity_ids: Optional[str] = None,
    query_type: Optional[str] = None,
    aggregate: Optional[bool] = None,
    analysis_mode: Optional[str] = None,
    forecast_duration: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取实体的时序指标数据，支持range/instant查询、聚合计算和高级分析模式。

    参数获取：
    - domain/entity_set_name: umodel_search_entity_set
    - metric_domain_name/metric: umodel_list_data_set(data_set_types="metric_set")
    - entity_ids: umodel_get_entities（可选）
    分析模式：basic/cluster/forecast/anomaly_detection。
    重要说明：metric_domain_name 必须使用 list_data_set 返回的 name（如 aiops.metric.service），
    不能使用 domain（如 aiops），否则会出现 Metric_Set not found。
    示例：
        umodel_get_metrics(
            workspace="default-cms-xxx",
            domain="aiops",
            entity_set_name="aiops.service",
            metric_domain_name="aiops.metric.service",
            metric="error",
            regionId="cn-heyuan",
            from_time=1766506202,
            to_time=1766507462
        )
    参数：workspace、domain、entity_set_name、metric_domain_name、metric、regionId（必需）；
    entity_ids、query_type、aggregate、analysis_mode、forecast_duration、from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_get_metrics",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "metric_domain_name": metric_domain_name,
                "metric": metric,
                "regionId": regionId,
                "entity_ids": entity_ids,
                "query_type": query_type,
                "aggregate": aggregate,
                "analysis_mode": analysis_mode,
                "forecast_duration": forecast_duration,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_get_relation_metrics(
    workspace: str,
    src_domain: str,
    src_entity_set_name: str,
    src_entity_ids: str,
    relation_type: str,
    direction: str,
    metric_set_domain: str,
    metric_set_name: str,
    metric: str,
    regionId: str,
    dest_domain: Optional[str] = None,
    dest_entity_set_name: Optional[str] = None,
    dest_entity_ids: Optional[str] = None,
    query_type: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取实体间关系级别的指标数据，如服务调用延迟、吞吐量等。

    参数获取：
    - src_domain/src_entity_set_name: umodel_search_entity_set
    - relation_type: umodel_list_related_entity_set
    - src_entity_ids: umodel_get_entities（必填）
    - metric_set_domain/metric_set_name/metric: umodel_list_data_set(data_set_types="metric_set")
    参数：workspace、src_domain、src_entity_set_name、src_entity_ids、relation_type、direction、metric_set_domain、metric_set_name、metric、regionId（必需）；
    dest_domain、dest_entity_set_name、dest_entity_ids、query_type、from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_get_relation_metrics",
        _clean_args(
            {
                "workspace": workspace,
                "src_domain": src_domain,
                "src_entity_set_name": src_entity_set_name,
                "src_entity_ids": src_entity_ids,
                "relation_type": relation_type,
                "direction": direction,
                "metric_set_domain": metric_set_domain,
                "metric_set_name": metric_set_name,
                "metric": metric,
                "regionId": regionId,
                "dest_domain": dest_domain,
                "dest_entity_set_name": dest_entity_set_name,
                "dest_entity_ids": dest_entity_ids,
                "query_type": query_type,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_get_logs(
    workspace: str,
    domain: str,
    entity_set_name: str,
    log_set_name: str,
    log_set_domain: str,
    regionId: str,
    entity_ids: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取实体相关的日志数据，用于故障诊断、性能分析、审计等场景。

    参数获取：domain/entity_set_name通过umodel_search_entity_set；log_set_domain/log_set_name通过umodel_list_data_set(data_set_types="log_set")；
    entity_ids可通过umodel_get_entities（可选）。
    参数：workspace、domain、entity_set_name、log_set_name、log_set_domain、regionId（必需）；entity_ids、from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_get_logs",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "log_set_name": log_set_name,
                "log_set_domain": log_set_domain,
                "regionId": regionId,
                "entity_ids": entity_ids,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_get_events(
    workspace: str,
    domain: str,
    entity_set_name: str,
    event_set_domain: str,
    event_set_name: str,
    regionId: str,
    entity_ids: Optional[str] = None,
    limit: Optional[int] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取指定实体集的事件数据。

    参数获取：event_set_domain/event_set_name可来自umodel_list_data_set(data_set_types="event_set")或默认"default"/"default.event.common"。
    参数：workspace、domain、entity_set_name、event_set_domain、event_set_name、regionId（必需）；entity_ids、limit、from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_get_events",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "event_set_domain": event_set_domain,
                "event_set_name": event_set_name,
                "regionId": regionId,
                "entity_ids": entity_ids,
                "limit": limit,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_search_traces(
    workspace: str,
    domain: str,
    entity_set_name: str,
    trace_set_domain: str,
    trace_set_name: str,
    regionId: str,
    entity_ids: Optional[str] = None,
    min_duration_ms: Optional[float] = None,
    max_duration_ms: Optional[float] = None,
    has_error: Optional[bool] = None,
    limit: Optional[int] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """基于过滤条件搜索trace并返回摘要信息。

    参数获取：domain/entity_set_name通过umodel_search_entity_set；trace_set_domain/trace_set_name通过umodel_list_data_set(data_set_types="trace_set")；
    entity_ids可选；过滤条件包括min_duration_ms/has_error/max_duration_ms。
    参数：workspace、domain、entity_set_name、trace_set_domain、trace_set_name、regionId（必需）；
    entity_ids、min_duration_ms、max_duration_ms、has_error、limit、from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_search_traces",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "trace_set_domain": trace_set_domain,
                "trace_set_name": trace_set_name,
                "regionId": regionId,
                "entity_ids": entity_ids,
                "min_duration_ms": min_duration_ms,
                "max_duration_ms": max_duration_ms,
                "has_error": has_error,
                "limit": limit,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_get_traces(
    workspace: str,
    domain: str,
    entity_set_name: str,
    trace_set_domain: str,
    trace_set_name: str,
    trace_ids: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取指定trace ID的详细trace数据，包括span、独占耗时与元数据。

    参数获取：trace_ids通常来自umodel_search_traces输出；trace_set_domain/trace_set_name通过umodel_list_data_set(data_set_types="trace_set")。
    输出字段：duration_ms、exclusive_duration_ms。
    参数：workspace、domain、entity_set_name、trace_set_domain、trace_set_name、trace_ids、regionId（必需）；from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_get_traces",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "trace_set_domain": trace_set_domain,
                "trace_set_name": trace_set_name,
                "trace_ids": trace_ids,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def umodel_get_profiles(
    workspace: str,
    domain: str,
    entity_set_name: str,
    profile_set_domain: str,
    profile_set_name: str,
    entity_ids: str,
    regionId: str,
    limit: Optional[int] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """获取指定实体集的性能剖析数据。

    参数获取：profile_set_domain/profile_set_name通过umodel_list_data_set(data_set_types="profile_set")；
    entity_ids必须指定（数据量大）。
    参数：workspace、domain、entity_set_name、profile_set_domain、profile_set_name、entity_ids、regionId（必需）；limit、from_time、to_time（可选）。
    """
    return await _call_tool(
        "umodel_get_profiles",
        _clean_args(
            {
                "workspace": workspace,
                "domain": domain,
                "entity_set_name": entity_set_name,
                "profile_set_domain": profile_set_domain,
                "profile_set_name": profile_set_name,
                "entity_ids": entity_ids,
                "regionId": regionId,
                "limit": limit,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def sls_text_to_sql(text: str, project: str, logStore: str, regionId: str) -> str:
    """将自然语言转换为SLS查询语句。

    功能概述：当用户有明确的logstore查询需求，必须优先使用本工具生成查询语句。
    使用限制：仅支持SLS查询语句，不支持其他数据库SQL；生成的是语句而非结果。
    最佳实践：描述中不要包含项目或日志库名称；可指定时间范围。
    """
    return await _call_tool(
        "sls_text_to_sql",
        {"text": text, "project": project, "logStore": logStore, "regionId": regionId},
    )


@tool
async def sls_execute_sql(
    project: str,
    logStore: str,
    query: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    reverse: Optional[bool] = None,
) -> str:
    """执行SLS日志查询。

    功能概述：在指定SLS项目与日志库执行查询并返回结果。
    使用要求：若上下文未提供SQL，必须先用sls_text_to_sql生成语句。
    时间范围：支持Unix时间戳（秒/毫秒）或相对时间表达式。
    """
    return await _call_tool(
        "sls_execute_sql",
        _clean_args(
            {
                "project": project,
                "logStore": logStore,
                "query": query,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
                "limit": limit,
                "offset": offset,
                "reverse": reverse,
            }
        ),
    )


@tool
async def sls_get_context_logs(
    project: str,
    logStore: str,
    pack_id: str,
    pack_meta: str,
    regionId: str,
    back_lines: Optional[int] = None,
    forward_lines: Optional[int] = None,
) -> str:
    """查询指定日志前后的上下文日志。

    功能概述：根据pack_id与pack_meta获取起始日志前后上下文日志（前后一天范围由服务限制）。
    获取方式：先用sls_execute_sql并在语句末尾加 |with_pack_meta 获取__pack_id__/__pack_meta__。
    参数说明：back_lines/forward_lines范围0~100且至少一个>0。
    """
    return await _call_tool(
        "sls_get_context_logs",
        _clean_args(
            {
                "project": project,
                "logStore": logStore,
                "pack_id": pack_id,
                "pack_meta": pack_meta,
                "regionId": regionId,
                "back_lines": back_lines,
                "forward_lines": forward_lines,
            }
        ),
    )


@tool
async def sls_log_explore(
    project: str,
    logStore: str,
    logField: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    filter_query: Optional[str] = None,
    groupField: Optional[str] = None,
) -> str:
    """查看日志库日志数据聚合分析结果，提供日志数据概览信息。

    功能概述：给出日志模板与各模板数量分布。
    使用场景：查看日志库概览信息与数据分布。
    """
    return await _call_tool(
        "sls_log_explore",
        _clean_args(
            {
                "project": project,
                "logStore": logStore,
                "logField": logField,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
                "filter_query": filter_query,
                "groupField": groupField,
            }
        ),
    )


@tool
async def sls_log_compare(
    project: str,
    logStore: str,
    logField: str,
    regionId: str,
    test_from_time: Optional[str] = None,
    test_to_time: Optional[str] = None,
    control_from_time: Optional[str] = None,
    control_to_time: Optional[str] = None,
    filter_query: Optional[str] = None,
    groupField: Optional[str] = None,
) -> str:
    """查看日志库在两个时间范围内的日志分布对比结果。

    功能概述：对比实验组与对照组时间范围内日志分布差异。
    使用场景：发布前后、昨日/今日日志变化分析。
    """
    return await _call_tool(
        "sls_log_compare",
        _clean_args(
            {
                "project": project,
                "logStore": logStore,
                "logField": logField,
                "regionId": regionId,
                "test_from_time": test_from_time,
                "test_to_time": test_to_time,
                "control_from_time": control_from_time,
                "control_to_time": control_to_time,
                "filter_query": filter_query,
                "groupField": groupField,
            }
        ),
    )


@tool
async def sls_list_projects(
    regionId: str,
    projectName: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """列出阿里云日志服务中的所有项目。

    功能概述：列出指定区域中的SLS项目，支持按名称模糊搜索。
    返回数据：project_name/description/region_id。
    """
    return await _call_tool(
        "sls_list_projects",
        _clean_args({"regionId": regionId, "projectName": projectName, "limit": limit}),
    )


@tool
async def sls_execute_spl(
    query: str,
    workspace: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """执行原生SPL查询语句。

    功能概述：为高级用户提供最大灵活性与复杂分析能力。
    注意事项：需要了解SPL语法，复杂查询可能消耗较多资源。
    """
    return await _call_tool(
        "sls_execute_spl",
        _clean_args(
            {
                "query": query,
                "workspace": workspace,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )


@tool
async def sls_list_logstores(
    project: str,
    regionId: str,
    logStore: Optional[str] = None,
    limit: Optional[int] = None,
    isMetricStore: Optional[bool] = None,
) -> str:
    """列出SLS项目中的日志库。

    功能概述：列出项目内日志库，支持名称模糊搜索。
    指标库：如需指标库请将isMetricStore设为True。
    """
    return await _call_tool(
        "sls_list_logstores",
        _clean_args(
            {
                "project": project,
                "regionId": regionId,
                "logStore": logStore,
                "limit": limit,
                "isMetricStore": isMetricStore,
            }
        ),
    )


@tool
async def cms_text_to_promql(
    text: str, project: str, metricStore: str, regionId: str
) -> str:
    """将自然语言转换为PromQL查询语句。

    功能概述：将自然语言描述转换为有效PromQL语句。
    使用限制：仅生成查询语句，不返回查询结果。
    """
    return await _call_tool(
        "cms_text_to_promql",
        {
            "text": text,
            "project": project,
            "metricStore": metricStore,
            "regionId": regionId,
        },
    )


@tool
async def cms_execute_promql(
    project: str,
    metricStore: str,
    query: str,
    regionId: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
) -> str:
    """执行PromQL指标查询。

    功能概述：在指定SLS项目与指标库执行PromQL查询并返回时序数据。
    时间范围：支持Unix时间戳（秒/毫秒）或相对时间表达式。
    """
    return await _call_tool(
        "cms_execute_promql",
        _clean_args(
            {
                "project": project,
                "metricStore": metricStore,
                "query": query,
                "regionId": regionId,
                "from_time": from_time,
                "to_time": to_time,
            }
        ),
    )
