import os
import sys
import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.multimodal_data import (
	get_logs_offline,
	get_trace_values_offline,
	time_series_anomaly_detection,
	trace_anomaly_detection,
	log_anomaly_detection,
	query_system_information,
	classifier,
)


def _df_to_json_records(df: pd.DataFrame, limit: int) -> str:
	if df.empty:
		return json.dumps({"count": 0, "limited_count": 0, "records": []})

	limited_df = df.head(limit)
	records = limited_df.to_dict(orient="records")
	return json.dumps(
		{
			"count": int(df.shape[0]),
			"limited_count": int(limited_df.shape[0]),
			"records": records,
		},
		ensure_ascii=False,
		default=str,
	)


@tool
def get_logs_timeseries(
	start_time: Annotated[str, "Start time, e.g. 2025-06-05T23:24:13Z"],
	end_time: Annotated[str, "End time, e.g. 2025-06-05T23:24:13Z"],
	limit: Annotated[int, "Max records to return"] = 200,
) -> str:
	"""
	Get time-ordered logs from offline parquet data.

	Returns:
		JSON string with total count and limited records.
	"""
	logs_df = get_logs_offline(start_time, end_time)
	if logs_df.empty:
		return "No logs found for the specified time period"

	return _df_to_json_records(logs_df, limit)


@tool
def get_traces_timeseries(
	start_time: Annotated[str, "Start time, e.g. 2025-06-05T23:24:13Z"],
	end_time: Annotated[str, "End time, e.g. 2025-06-05T23:24:13Z"],
	limit: Annotated[int, "Max records to return"] = 200,
) -> str:
	"""
	Get trace summary (avg duration & count per service/pod) from offline parquet data.

	Returns:
		JSON string with total count and limited records.
	"""
	trace_df = get_trace_values_offline(start_time, end_time)
	if trace_df.empty:
		return "No trace data found"

	return _df_to_json_records(trace_df, limit)


@tool
def search_metrics(
	start_time: str,
	end_time: str,
	metric_name: str = "all",
	instance: str = "all",
) -> str:
	"""
	Search and analyze metric anomalies.

	Args:
		start_time: Start time in format like 2025-06-05T23:24:13Z
		end_time: End time in format like 2025-06-05T23:24:13Z
		metric_name: Name of the metric to check. Defaults to 'all'.
		instance: Name of the instance (service/pod/node) to check. Defaults to 'all'.
	"""
	try:
		return time_series_anomaly_detection(
			start_time, end_time, metric_name, instance
		)
	except Exception as e:
		return f"Error executing time_series_anomaly_detection: {str(e)}"


@tool
def search_traces(start_time: str, end_time: str) -> str:
	"""
	Detect anomalies in traces.

	Args:
		start_time: Start time in format like 2025-06-05T23:24:13Z
		end_time: End time in format like 2025-06-05T23:24:13Z
	"""
	try:
		return trace_anomaly_detection(start_time, end_time)
	except Exception as e:
		return f"Error executing trace_anomaly_detection: {str(e)}"


@tool
def search_logs(start_time: str, end_time: str) -> str:
	"""
	Detect anomalies in logs.

	Args:
		start_time: Start time in format like 2025-06-05T23:24:13Z
		end_time: End time in format like 2025-06-05T23:24:13Z
	"""
	try:
		return log_anomaly_detection(start_time, end_time)
	except Exception as e:
		return f"Error executing log_anomaly_detection: {str(e)}"


@tool
def get_system_info() -> str:
	"""
	Retrieve system topology and configuration information.
	Use this tool to understand the static architecture of the system, including:
	- Service call relationships and dependencies (Call Graph).
	- Deployment information (which services run on which nodes).
	- Configuration details of components (e.g., database versions, resource limits).
	Input is a natural language question about the system structure or topology.
	"""
	return query_system_information()


@tool
def analyze_fault_type(start_time: str, end_time: str) -> str:
	"""
	Analyze anomaly detection results to determine the main fault type.
	This tool uses a classifier to identify the likely failure category (e.g., Pod Memory, Network Delay) based on metric patterns.

	Args:
		start_time: Start time in format like 2025-06-05T23:24:13Z
		end_time: End time in format like 2025-06-05T23:24:13Z
	"""
	try:
		return classifier(start_time, end_time)
	except Exception as e:
		return f"Error executing classifier: {str(e)}"
