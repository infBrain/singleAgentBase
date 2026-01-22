import os
import sys
import json
from typing import Annotated

import pandas as pd
from langchain_core.tools import tool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.multimodal_data import get_logs_offline, get_trace_values_offline


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
