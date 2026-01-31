import datetime
import time
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from mcp import ClientSession
from mcp.types import CallToolResult
from pandas import Timestamp

_MCP_SESSION: Optional[ClientSession] = None


def set_mcp_session(session: ClientSession) -> None:
    """Set global MCP session for @tool wrappers."""
    global _MCP_SESSION
    _MCP_SESSION = session


def clear_mcp_session() -> None:
    """Clear the global MCP session."""
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

def build_project_details(
    workspace, region, sls_project, logstore, metircstore, tracestore
):
    return f"""Your UModel workspace is '{workspace}' in region '{region}', and the SLS project is '{sls_project}'.
    The logstore is '{logstore}', the metricstore is '{metircstore}', the tracestore is '{tracestore}'.
    Use this information when configuring your data source connections.
    """

@tool
async def guide_intro() -> str:
    """Return guidance for root cause analysis (RCA) using MCP tools and workflow.

    Overview: Provides expectations, call order, parameter choices, and error handling tips for RCA tasks.
    Use case: Obtain a unified RCA workflow guide before executing root cause localization tasks.
    Notes: Returns guidance text only and does not call MCP APIs.
    """
    return """MCP RCA Workflow (trace-first with strict gating):
        1) Setup discovery:
        - list_workspace -> list_domains.
        - Find the entity set for services/pods/nodes via umodel_search_entity_set, then get entities via umodel_get_entities (only if needed).
        2) Trace-first:
        - umodel_list_data_set(...trace_set...) -> umodel_search_traces(has_error/min_duration_ms) -> umodel_get_traces.
        - If abnormal trace exists: walk root span, collect earliest abnormal entities as candidates.
        3) If trace is missing / no abnormal trace:
        - Use topology tools (umodel_get_neighbor_entities / umodel_list_related_entity_set) to derive entrypoints and call graph.
        - Expand from entrypoints layer by layer with stop-loss (N visited, K children).
        - Metrics gate first: umodel_get_golden_metrics (if available) else umodel_get_metrics(anomaly_detection).
        - Logs confirmation: umodel_list_data_set(...log_set...) -> umodel_get_logs OR sls_text_to_sql + sls_execute_sql.
        4) After candidates emerge, optionally pull related traces again to confirm abnormal calls.
        5) Validate with topology direction and output up to 3 root causes mapped to {instance_type}.
        Rules: never fabricate workspace/domain/set/metric/log_set; always discover first; if empty, narrow time window or stop.
        """

@tool
async def introduction() -> str:
    """Get an introduction and usage guide for the Alibaba Cloud Observability MCP Server.

    Overview: Returns service overview, core capabilities, and usage limits.
    Use case: Learn capabilities and prerequisites on first access.
    Notes: No parameters required; returned info includes prerequisites for tools.
    """
    return await _call_tool("introduction", {})

@tool
async def list_workspace(regionId: str) -> str:
    """List available CMS workspaces.

    Overview: Get CMS workspaces in the specified region.
    Args: regionId (Aliyun region ID, e.g. "cn-hangzhou").
    Returns: Workspace info dictionary (workspaces/total_count/region).
    """
    return await _call_tool("list_workspace", {"regionId": regionId})


@tool
async def list_domains(workspace: str, regionId: str) -> str:
    """List all available entity domains.

    Overview: Get all available entity domains.
    Use case: Discover supported domains and choose the correct domain.
    Returns: Each item includes __domain__ (name) and cnt (entity count).
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
    """Search entity sets with full-text search and relevance sorting.

    Overview: Search UModel metadata for entity set definitions; returns domain/name/display_name.
    Use case: Entity set discovery and metadata exploration.
    Args: search_text/workspace/regionId (required); domain/entity_set_name/limit (optional).
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
    """List available data sets for an entity to supply parameters for PaaS tools.

    Overview: Get available data sets for a given domain and entity set.
    Args: data_set_types supports metric_set/log_set/event_set/trace_set/entity_set_name.
    Dependencies: Used by umodel_get_metrics/logs/events/traces.
    Example:
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
    """List entity sets related to the specified entity set.

    Overview: Discover entity sets that have relationship definitions with the source set.
    Features: Supports direction and relation_type filters; returns metadata-level definitions.
    Args: workspace/domain/entity_set_name/regionId (required); relation_type/direction/detail (optional).
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
    """PaaS API tool to get entity information.

    Overview: Retrieve basic entity info with pagination and exact ID queries.
    Scope: Use this for basic entity info; for fuzzy search use umodel_search_entities.
    Args: workspace/domain/entity_set_name/regionId (required); entity_ids/from_time/to_time/limit (optional).
    Example:
        # Get entities and use returned __entity_id__ for follow-up queries
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
    """Search entities by keyword.

    Overview: Fuzzy and full-text search within the specified entity set.
    Features: Supports limit control and flexible filtering.
    Args: workspace/domain/entity_set_name/search_text/regionId (required); from_time/to_time/limit (optional).
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
    """Get neighbor entities for a specified entity.

    Overview: Find neighboring entities based on relationships.
    Use cases: Dependency analysis, relationship discovery, impact assessment, topology building.
    Args: workspace/domain/entity_set_name/entity_id/regionId (required); from_time/to_time/limit (optional).
    Example:
        # entity_id from umodel_get_entities (__entity_id__)
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
    """Get golden metrics (key performance indicators) for entities.

    Parameter lookup: domain/entity_set_name from umodel_search_entity_set; entity_ids from umodel_get_entities.
    Args: workspace/domain/entity_set_name/regionId (required); entity_ids/from_time/to_time (optional).
    Example:
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
    """Get time-series metrics for entities; supports range/instant, aggregation, and analysis modes.

    Parameter lookup:
    - domain/entity_set_name: umodel_search_entity_set
    - metric_domain_name/metric: umodel_list_data_set(data_set_types="metric_set")
    - entity_ids: umodel_get_entities (optional)
    Analysis modes: basic/cluster/forecast/anomaly_detection.
    Important: metric_domain_name must be the name returned by list_data_set (e.g. aiops.metric.service),
    not the domain (e.g. aiops), otherwise Metric_Set not found.
    Example:
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
    Args: workspace/domain/entity_set_name/metric_domain_name/metric/regionId (required);
    entity_ids/query_type/aggregate/analysis_mode/forecast_duration/from_time/to_time (optional).
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
    """Get relation-level metrics between entities, e.g., call latency and throughput.

    Parameter lookup:
    - src_domain/src_entity_set_name: umodel_search_entity_set
    - relation_type: umodel_list_related_entity_set
    - src_entity_ids: umodel_get_entities (required)
    - metric_set_domain/metric_set_name/metric: umodel_list_data_set(data_set_types="metric_set")
    Args: workspace/src_domain/src_entity_set_name/src_entity_ids/relation_type/direction/
    metric_set_domain/metric_set_name/metric/regionId (required);
    dest_domain/dest_entity_set_name/dest_entity_ids/query_type/from_time/to_time (optional).
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
    """Get entity-related logs for diagnostics, performance analysis, and auditing.

    Parameter lookup: domain/entity_set_name from umodel_search_entity_set;
    log_set_domain/log_set_name from umodel_list_data_set(data_set_types="log_set");
    entity_ids from umodel_get_entities (optional).
    Args: workspace/domain/entity_set_name/log_set_name/log_set_domain/regionId (required);
    entity_ids/from_time/to_time (optional).
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
    """Get event data for a specified entity set.

    Parameter lookup: event_set_domain/event_set_name from umodel_list_data_set(data_set_types="event_set")
    or defaults "default"/"default.event.common".
    Args: workspace/domain/entity_set_name/event_set_domain/event_set_name/regionId (required);
    entity_ids/limit/from_time/to_time (optional).
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
    """Search traces with filters and return summaries.

    Parameter lookup: domain/entity_set_name from umodel_search_entity_set;
    trace_set_domain/trace_set_name from umodel_list_data_set(data_set_types="trace_set");
    entity_ids optional; filters include min_duration_ms/has_error/max_duration_ms.
    Args: workspace/domain/entity_set_name/trace_set_domain/trace_set_name/regionId (required);
    entity_ids/min_duration_ms/max_duration_ms/has_error/limit/from_time/to_time (optional).
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
    """Get detailed trace data for specified trace IDs, including spans and metadata.

    Parameter lookup: trace_ids usually from umodel_search_traces; trace_set_domain/trace_set_name from
    umodel_list_data_set(data_set_types="trace_set").
    Output fields: duration_ms, exclusive_duration_ms.
    Args: workspace/domain/entity_set_name/trace_set_domain/trace_set_name/trace_ids/regionId (required);
    from_time/to_time (optional).
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
    """Get performance profiling data for a specified entity set.

    Parameter lookup: profile_set_domain/profile_set_name from umodel_list_data_set(data_set_types="profile_set");
    entity_ids is required (large data volume).
    Args: workspace/domain/entity_set_name/profile_set_domain/profile_set_name/entity_ids/regionId (required);
    limit/from_time/to_time (optional).
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
    """Convert natural language to an SLS query.

    Overview: When a logstore query is needed, use this tool first to generate the query.
    Limits: Only SLS query syntax is supported; returns a query string, not results.
    Best practice: Do not include project or logstore names in the text; specify time range if needed.
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
    """Execute an SLS log query.

    Overview: Run a query in the specified SLS project and logstore.
    Requirement: If no SQL is provided, generate it with sls_text_to_sql first.
    Time range: Supports Unix timestamps (seconds/ms) or relative expressions.
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
    """Query context logs before and after a specific log entry.

    Overview: Use pack_id/pack_meta to fetch surrounding context (up to one day before/after).
    How to get pack fields: Run sls_execute_sql and append |with_pack_meta to get __pack_id__/__pack_meta__.
    Args: back_lines/forward_lines range 0~100 and at least one must be > 0.
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
    """Explore aggregated log patterns to provide a log overview.

    Overview: Returns log templates and their distribution.
    Use case: View logstore overview and data distribution.
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
    """Compare log distributions across two time ranges.

    Overview: Compare log distribution differences between test and control windows.
    Use case: Before/after release, yesterday vs today changes.
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
    """List all projects in Alibaba Cloud Log Service.

    Overview: List SLS projects in the specified region; supports fuzzy name search.
    Returns: project_name/description/region_id.
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
    """Execute a raw SPL query.

    Overview: Provides maximum flexibility and advanced analysis.
    Notes: Requires SPL knowledge; complex queries may consume more resources.
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
    """List logstores in an SLS project.

    Overview: List logstores with optional fuzzy name search.
    Metric stores: Set isMetricStore=True to list metric stores.
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
    """Convert natural language to a PromQL query.

    Overview: Convert a natural language description into a valid PromQL expression.
    Limit: Generates query only, not results.
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
    """Execute a PromQL metrics query.

    Overview: Run PromQL against a metric store and return time-series data.
    Time range: Supports Unix timestamps (seconds/ms) or relative expressions.
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
