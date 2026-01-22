from typing import Any, Dict, Optional, List

from mcp import ClientSession
from mcp.types import CallToolResult


async def _call_tool(
	session: ClientSession, name: str, arguments: Dict[str, Any]
) -> str:
	"""Generic MCP call wrapper for query-style tools."""
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


async def list_workspace(session: ClientSession, regionId: str) -> str:
	return await _call_tool(session, "list_workspace", {"regionId": regionId})


async def list_domains(session: ClientSession, workspace: str, regionId: str) -> str:
	return await _call_tool(
		session, "list_domains", {"workspace": workspace, "regionId": regionId}
	)


async def introduction(session: ClientSession) -> str:
	return await _call_tool(session, "introduction", {})


async def umodel_search_entity_set(
	session: ClientSession, workspace: str, search_text: str, regionId: str
) -> str:
	return await _call_tool(
		session,
		"umodel_search_entity_set",
		{"workspace": workspace, "search_text": search_text, "regionId": regionId},
	)


async def umodel_list_related_entity_set(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_list_related_entity_set",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"regionId": regionId,
		},
	)


async def umodel_list_data_set(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	regionId: str,
	data_set_types: Optional[List[str]] = None,
) -> str:
	arguments: Dict[str, Any] = {
		"workspace": workspace,
		"domain": domain,
		"entity_set_name": entity_set_name,
		"regionId": regionId,
	}
	if data_set_types:
		arguments["data_set_types"] = data_set_types

	return await _call_tool(session, "umodel_list_data_set", arguments)


async def umodel_get_entities(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_entities",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"regionId": regionId,
		},
	)


async def umodel_get_neighbor_entities(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	entity_ids: List[str],
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_neighbor_entities",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"entity_ids": entity_ids,
			"regionId": regionId,
		},
	)


async def umodel_search_entities(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	search_conditions: Dict[str, Any],
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_search_entities",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"search_conditions": search_conditions,
			"regionId": regionId,
		},
	)


async def umodel_get_golden_metrics(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_golden_metrics",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"regionId": regionId,
		},
	)


async def umodel_get_metrics(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	metric_domain_name: str,
	metric: str,
	regionId: str,
	analysis_mode: Optional[str] = None,
	forecast_duration: Optional[int] = None,
) -> str:
	arguments: Dict[str, Any] = {
		"workspace": workspace,
		"domain": domain,
		"entity_set_name": entity_set_name,
		"metric_domain_name": metric_domain_name,
		"metric": metric,
		"regionId": regionId,
	}
	if analysis_mode:
		arguments["analysis_mode"] = analysis_mode
	if forecast_duration is not None:
		arguments["forecast_duration"] = forecast_duration

	return await _call_tool(session, "umodel_get_metrics", arguments)


async def umodel_get_relation_metrics(
	session: ClientSession,
	workspace: str,
	src_domain: str,
	src_entity_set_name: str,
	src_entity_ids: List[str],
	relation_type: str,
	direction: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_relation_metrics",
		{
			"workspace": workspace,
			"src_domain": src_domain,
			"src_entity_set_name": src_entity_set_name,
			"src_entity_ids": src_entity_ids,
			"relation_type": relation_type,
			"direction": direction,
			"regionId": regionId,
		},
	)


async def umodel_get_logs(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	log_set_name: str,
	log_set_domain: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_logs",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"log_set_name": log_set_name,
			"log_set_domain": log_set_domain,
			"regionId": regionId,
		},
	)


async def umodel_get_events(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	event_set_domain: str,
	event_set_name: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_events",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"event_set_domain": event_set_domain,
			"event_set_name": event_set_name,
			"regionId": regionId,
		},
	)


async def umodel_search_traces(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	trace_set_domain: str,
	trace_set_name: str,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_search_traces",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"trace_set_domain": trace_set_domain,
			"trace_set_name": trace_set_name,
			"regionId": regionId,
		},
	)


async def umodel_get_traces(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	trace_set_domain: str,
	trace_set_name: str,
	trace_ids: List[str],
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_traces",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"trace_set_domain": trace_set_domain,
			"trace_set_name": trace_set_name,
			"trace_ids": trace_ids,
			"regionId": regionId,
		},
	)


async def umodel_get_profiles(
	session: ClientSession,
	workspace: str,
	domain: str,
	entity_set_name: str,
	profile_set_domain: str,
	profile_set_name: str,
	entity_ids: List[str],
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"umodel_get_profiles",
		{
			"workspace": workspace,
			"domain": domain,
			"entity_set_name": entity_set_name,
			"profile_set_domain": profile_set_domain,
			"profile_set_name": profile_set_name,
			"entity_ids": entity_ids,
			"regionId": regionId,
		},
	)


async def sls_text_to_sql(
	session: ClientSession, text: str, project: str, logStore: str, regionId: str
) -> str:
	return await _call_tool(
		session,
		"sls_text_to_sql",
		{"text": text, "project": project, "logStore": logStore, "regionId": regionId},
	)


async def sls_list_projects(
	session: ClientSession, regionId: str, projectName: Optional[str] = None
) -> str:
	arguments: Dict[str, Any] = {"regionId": regionId}
	if projectName:
		arguments["projectName"] = projectName

	return await _call_tool(session, "sls_list_projects", arguments)


async def sls_execute_sql(
	session: ClientSession,
	project: str,
	logStore: str,
	query: str,
	from_time: int,
	to_time: int,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"sls_execute_sql",
		{
			"project": project,
			"logStore": logStore,
			"query": query,
			"from_time": from_time,
			"to_time": to_time,
			"regionId": regionId,
		},
	)


async def sls_execute_spl(
	session: ClientSession, query: str, regionId: str
) -> str:
	return await _call_tool(
		session, "sls_execute_spl", {"query": query, "regionId": regionId}
	)


async def sls_list_logstores(
	session: ClientSession, project: str, regionId: str
) -> str:
	return await _call_tool(
		session,
		"sls_list_logstores",
		{"project": project, "regionId": regionId},
	)


async def cms_text_to_promql(
	session: ClientSession, text: str, project: str, metricStore: str, regionId: str
) -> str:
	return await _call_tool(
		session,
		"cms_text_to_promql",
		{
			"text": text,
			"project": project,
			"metricStore": metricStore,
			"regionId": regionId,
		},
	)


async def cms_execute_promql(
	session: ClientSession,
	project: str,
	metricStore: str,
	query: str,
	start_time: int,
	end_time: int,
	regionId: str,
) -> str:
	return await _call_tool(
		session,
		"cms_execute_promql",
		{
			"project": project,
			"metricStore": metricStore,
			"query": query,
			"start_time": start_time,
			"end_time": end_time,
			"regionId": regionId,
		},
	)
