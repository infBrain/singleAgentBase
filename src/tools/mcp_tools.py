from typing import Any, Dict, Optional

from langchain_core.tools import tool
from mcp import ClientSession
from mcp.types import CallToolResult

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
		raise RuntimeError("MCP session is not set. Call set_mcp_session(session) first.")
	return _MCP_SESSION


def _clean_args(arguments: Dict[str, Any]) -> Dict[str, Any]:
	return {key: value for key, value in arguments.items() if value is not None}


async def _call_tool(name: str, arguments: Dict[str, Any]) -> str:
	"""Generic MCP call wrapper for query-style tools."""
	session = _require_session()
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
	"""获取阿里云可观测性 MCP Server 的介绍和使用说明。

	功能概述：返回服务概述、核心能力与使用限制说明。
	使用场景：首次接入时了解服务能力与前提。
"""
	return await _call_tool("introduction", {})


@tool
async def list_workspace(regionId: str) -> str:
	"""列出可用的 CMS 工作空间。

	参数：regionId（必需）。
	返回：包含工作空间信息的列表。
"""
	return await _call_tool("list_workspace", {"regionId": regionId})


@tool
async def list_domains(workspace: str, regionId: str) -> str:
	"""列出所有可用的实体域。

	参数：workspace、regionId（必需）。
	返回：实体域列表。
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
	"""搜索实体集合（全文搜索并按相关度排序）。

	参数：workspace、search_text、regionId（必需）；domain、entity_set_name、limit（可选）。
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
	"""列出指定实体的可用数据集合。

	参数：workspace、domain、entity_set_name、regionId（必需）；data_set_types、from_time、to_time（可选）。
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
	"""获取实体信息的 PaaS API 工具。

	参数：workspace、domain、entity_set_name、regionId（必需）；entity_ids、from_time、to_time、limit（可选）。
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

	参数：workspace、domain、entity_set_name、entity_id、regionId（必需）；from_time、to_time、limit（可选）。
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

	参数：workspace、domain、entity_set_name、regionId（必需）；entity_ids、from_time、to_time（可选）。
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
	"""获取实体的时序指标数据，支持高级分析模式。

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
	"""获取实体间关系级别的指标数据。

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
	"""获取实体相关的日志数据。

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
	"""基于过滤条件搜索 trace 并返回摘要信息。

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
	"""获取指定 trace ID 的详细 trace 数据。

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
async def sls_text_to_sql(
	text: str, project: str, logStore: str, regionId: str
) -> str:
	"""将自然语言转换为 SLS 查询语句。

	参数：text、project、logStore、regionId（必需）。
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
	"""执行 SLS 日志查询。

	参数：project、logStore、query、regionId（必需）；from_time、to_time、limit、offset、reverse（可选）。
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

	参数：project、logStore、pack_id、pack_meta、regionId（必需）；back_lines、forward_lines（可选）。
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
	"""查看日志库日志数据概览与分布。

	参数：project、logStore、logField、regionId（必需）；from_time、to_time、filter_query、groupField（可选）。
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
	"""对比两个时间范围内的日志分布变化。

	参数：project、logStore、logField、regionId（必需）；test_from_time、test_to_time、control_from_time、control_to_time、filter_query、groupField（可选）。
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
	"""列出阿里云日志服务中的项目。

	参数：regionId（必需）；projectName、limit（可选）。
"""
	return await _call_tool(
		"sls_list_projects",
		_clean_args(
			{"regionId": regionId, "projectName": projectName, "limit": limit}
		),
	)


@tool
async def sls_execute_spl(
	query: str,
	workspace: str,
	regionId: str,
	from_time: Optional[str] = None,
	to_time: Optional[str] = None,
) -> str:
	"""执行原生 SPL 查询语句。

	参数：query、workspace、regionId（必需）；from_time、to_time（可选）。
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
	"""列出 SLS 项目中的日志库。

	参数：project、regionId（必需）；logStore、limit、isMetricStore（可选）。
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
	"""将自然语言转换为 PromQL 查询语句。

	参数：text、project、metricStore、regionId（必需）。
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
	"""执行 PromQL 指标查询。

	参数：project、metricStore、query、regionId（必需）；from_time、to_time（可选）。
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
