import os
import sys
import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.multimodal_data import (
    get_logs_offline,
    get_metric_values_offline,
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
def guide_intro() -> str:
    """Return guidance for traditional offline tool usage flow and norms.

    Overview: Recommends an order for offline data queries, anomaly detection, and fault type analysis.
    Use case: Get a unified workflow guide before starting offline RCA tasks.
    Notes: Returns guidance text only and does not execute queries.
    """
    return """Traditional RCA Workflow (trace-first with topology fallback):
		1) Confirm {{start_time}}/{{end_time}}/{{instance_type}}.
		2) Try traces first:
		- get_traces / detect_traces for error/slow traces in the window.
		- If traces abnormal: walk from root span, pick earliest abnormal hop as candidates.
		3) If traces missing or show no clear anomaly:
		- get_system_info to obtain topology/call graph and entrypoints.
		- From entrypoints, expand layer by layer; at each step use detect_metrics as the primary gate, and detect_logs as confirmation.
		- Only expand to a node’s top K downstream neighbors; stop after visiting N nodes.
		4) After at least one strong signal exists, call analyze_fault_type.
		5) Validate cause vs symptom using topology direction.
		6) Output up to 3 root causes mapped to {{instance_type}}.
		Rules: never guess parameters; if a modality has no data, skip it and continue.
		"""


@tool
def get_logs(
    start_time: Annotated[str, "Start time, e.g. 2025-06-05T23:24:13Z"],
    end_time: Annotated[str, "End time, e.g. 2025-06-05T23:24:13Z"],
    limit: Annotated[int, "Max records to return"] = 200,
) -> str:
    """
    Get time-ordered logs from offline parquet data.

    Returns:
            JSON string with total count and limited records.
    """
    print(f"[get_logs] called with start_time={start_time}, end_time={end_time}, limit={limit}")
    logs_df = get_logs_offline(start_time, end_time)
    if logs_df.empty:
        print("[get_logs] No logs found for the specified time period")
        return "No logs found for the specified time period"

    result = _df_to_json_records(logs_df, limit)
    print(f"[get_logs] returning {min(limit, logs_df.shape[0])} records (total: {logs_df.shape[0]})")
    return result


@tool
def get_traces(
    start_time: Annotated[str, "Start time, e.g. 2025-06-05T23:24:13Z"],
    end_time: Annotated[str, "End time, e.g. 2025-06-05T23:24:13Z"],
    limit: Annotated[int, "Max records to return"] = 200,
) -> str:
    """
    Get trace summary (avg duration & count per service/pod) from offline parquet data.

    Returns:
            JSON string with total count and limited records.
    """
    print(f"[get_traces] called with start_time={start_time}, end_time={end_time}, limit={limit}")
    trace_df = get_trace_values_offline(start_time, end_time)
    if trace_df.empty:
        print("[get_traces] No trace data found")
        return "No trace data found"

    result = _df_to_json_records(trace_df, limit)
    print(f"[get_traces] returning {min(limit, trace_df.shape[0])} records (total: {trace_df.shape[0]})")
    return result


@tool
def get_metrics(
    start_time: str,
    end_time: str,
	instance: str,
    metric_name: str = "all",
) -> str:
    """
    Get metric values from offline data.

    Args:
            start_time: Start time in format like 2025-06-05T23:24:13Z
            end_time: End time in format like 2025-06-05T23:24:13Z
            metric_name: Name of the metric to check. Defaults to 'all'.
            instance: Name of the instance (service/pod/node) to check.
    """
    try:
        print(f"[get_metrics] called with start_time={start_time}, end_time={end_time}, metric_name={metric_name}, instance={instance}")
        values = get_metric_values_offline(
            start_time=start_time,
            end_time=end_time,
            metric_name=metric_name,
            instance=instance,
        )
        if values is None or len(values) == 0:
            print("[get_metrics] No metric data found for the specified time period")
            return "No metric data found for the specified time period"
        limited_values = values[:200].tolist()
        print(f"[get_metrics] returning {len(limited_values)} values (total: {len(values)})")
        return json.dumps(
            {
                "count": int(len(values)),
                "limited_count": int(len(limited_values)),
                "values": limited_values,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        print(f"[get_metrics] Error: {str(e)}")
        return f"Error executing get_metric_values_offline: {str(e)}"


@tool
def detect_metrics(
    start_time: str,
    end_time: str,
    instance: str,
    metric_name: str = "all",
) -> str:
    """
    Search and analyze the instance‘s metric anomalies.

    Args:
            start_time: Start time in format like 2025-06-05T23:24:13Z
            end_time: End time in format like 2025-06-05T23:24:13Z
            metric_name: Name of the metric to check. Defaults to 'all'.
            instance: Name of the instance (service/pod/node) to check.
    """
    try:
        print(f"[detect_metrics] called with start_time={start_time}, end_time={end_time}, metric_name={metric_name}, instance={instance}")
        result = time_series_anomaly_detection(
            start_time, end_time, metric_name, instance
        )
        print(f"[detect_metrics] result: {str(result)[:200]}{'...' if len(str(result))>200 else ''}")
        return result
    except Exception as e:
        print(f"[detect_metrics] Error: {str(e)}")
        return f"Error executing time_series_anomaly_detection: {str(e)}"


@tool
def detect_traces(start_time: str, end_time: str) -> str:
    """
    Detect anomalies in traces.

    Args:
            start_time: Start time in format like 2025-06-05T23:24:13Z
            end_time: End time in format like 2025-06-05T23:24:13Z
    """
    try:
        print(f"[detect_traces] called with start_time={start_time}, end_time={end_time}")
        result = trace_anomaly_detection(start_time, end_time)
        print(f"[detect_traces] result: {str(result)[:200]}{'...' if len(str(result))>200 else ''}")
        return result
    except Exception as e:
        print(f"[detect_traces] Error: {str(e)}")
        return f"Error executing trace_anomaly_detection: {str(e)}"


@tool
def detect_logs(start_time: str, end_time: str) -> str:
    """
    Detect anomalies in logs.

    Args:
            start_time: Start time in format like 2025-06-05T23:24:13Z
            end_time: End time in format like 2025-06-05T23:24:13Z
    """
    try:
        print(f"[detect_logs] called with start_time={start_time}, end_time={end_time}")
        result = log_anomaly_detection(start_time, end_time)
        print(f"[detect_logs] result: {str(result)[:200]}{'...' if len(str(result))>200 else ''}")
        return result
    except Exception as e:
        print(f"[detect_logs] Error: {str(e)}")
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
    print("[get_system_info] called")
    result = query_system_information()
    print(f"[get_system_info] result: {str(result)[:200]}{'...' if len(str(result))>200 else ''}")
    return result


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
        print(f"[analyze_fault_type] called with start_time={start_time}, end_time={end_time}")
        result = classifier(start_time, end_time)
        print(f"[analyze_fault_type] result: {str(result)[:200]}{'...' if len(str(result))>200 else ''}")
        return result
    except Exception as e:
        print(f"[analyze_fault_type] Error: {str(e)}")
        return f"Error executing classifier: {str(e)}"
