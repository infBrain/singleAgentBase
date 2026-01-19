import sys

# sys.path.append("./")
# sys.path.append("../")
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trace
from typing import Annotated, List, Tuple

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
import json


from tools.utils import (
    ML_METRICS,
    _get_date_range,
    _get_instance_type,
    _convert_to_utc,
    _get_metric_type,
    _parse_and_convert_time,
    SIGMA_METRICS,
    THRESHOLD_METRICS,
    _detect_anomalies_sigma,
    _detect_anomalies_threshold,
    _generate_anomaly_description,
)
from tools.utils import (
    apm_metric_names,
    infra_node_metric_names,
    infra_pod_metric_names,
    infra_tidb_metric_names,
    tidb_pd_metric_names,
    tidb_tikv_metric_names,
)
from tools.llm_chat import chat
from tools.fcvae_model import FCVAE
import matplotlib.pyplot as plt
import torch

WINDOW_SIZE = 256

# apm_metric_names = ["client_error","client_error_ratio","error","error_ratio","request","response","rrt","rrt_max","server_error","server_error_ratio","timeout"]
# infra_pod_metric_names = ["pod_cpu_usage", "pod_fs_reads_bytes", "pod_fs_writes_bytes", "pod_memory_working_set_bytes", "pod_network_receive_bytes", "pod_network_receive_packets", "pod_network_transmit_bytes", "pod_network_transmit_packets", "pod_processes", "node_cpu_usage_rate", "node_disk_read_bytes_total", "node_disk_read_time_seconds_total", "node_disk_write_time_seconds_total", "node_disk_written_bytes_total", "node_filesystem_free_bytes", "node_filesystem_size_bytes", "node_filesystem_usage_rate", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes", "node_memory_usage_rate", "node_network_receive_bytes_total", "node_network_receive_packets_total", "node_network_transmit_bytes_total", "node_network_transmit_packets_total", "node_sockstat_TCP_inuse"]
# infra_node_metric_names = ["node_cpu_usage_rate", "node_disk_read_bytes_total", "node_disk_read_time_seconds_total", "node_disk_write_time_seconds_total", "node_disk_written_bytes_total", "node_filesystem_free_bytes", "node_filesystem_size_bytes", "node_filesystem_usage_rate", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes", "node_memory_usage_rate", "node_network_receive_bytes_total", "node_network_receive_packets_total", "node_network_transmit_bytes_total", "node_network_transmit_packets_total", "node_sockstat_TCP_inuse"]


def get_metric_values_offline(
    start_time: Annotated[str, "start time of the anomaly"],
    end_time: Annotated[str, "end time of the anomaly"],
    metric_name: Annotated[str, "name of the metric"],
    instance: Annotated[str, "name of the pod, service or node"],
) -> pd.DataFrame:
    """
    Collect metric data from the data folder

    Args:
        start_time: Supports two formats:
                   1. Beijing time format (e.g., '2024-09-12 20:08:19')
                   2. UTC format (e.g., '2024-09-12T12:08:19Z')
        end_time: Same formats as start_time
        metric_name: Name of the metric
        instance: Instance name (pod name, service name, or node name)

    Returns:
        DataFrame containing metric data
    """
    if instance.startswith("tidb-"):
        # 处理tidb类型
        prefix = instance.split("-")[1] + "_"
        if not metric_name.startswith(prefix):
            init_metric_name = metric_name = prefix + metric_name
        else:
            init_metric_name = metric_name

    # Get metric type and instance type
    metric_type = _get_metric_type(metric_name)
    instance_type = _get_instance_type(instance)

    # Validate metric type and instance type match
    if metric_type == "infra_node" and instance_type != "node":
        raise Exception(
            f"Node metric ({metric_name}) cannot be used with non-node instance ({instance})"
        )
    elif metric_type == "infra_pod" and instance_type != "pod":
        raise Exception(
            f"Pod metric ({metric_name}) cannot be used with non-pod instance ({instance})"
        )
    elif (
        metric_type == "infra_tidb"
        or metric_type == "tidb_pd"
        or metric_type == "tidb_tikv"
    ) and instance_type != "pod":
        raise Exception(
            f"TiDB metric ({metric_name}) cannot be used with non-pod instance ({instance})"
        )

    # Parse and convert times to UTC
    start_utc = _parse_and_convert_time(start_time)
    end_utc = _parse_and_convert_time(end_time)

    # Get date range
    dates = _get_date_range(start_utc, end_utc)

    all_data = []
    # base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )

    for date in dates:
        # date_folder = f"metric-parquet_{date}"
        date_folder = f"{date.replace('-', '')}/metric-parquet"
        date_path = os.path.join(base_path, date_folder)

        if not os.path.exists(date_path):
            continue

        # Determine file path based on metric type and instance type
        if metric_type == "apm":
            if instance_type == "pod":
                path = os.path.join(
                    date_path, "apm", "pod", f"pod_{instance}_{date}.parquet"
                )
                if os.path.exists(path):
                    # Read parquet file, select only needed columns
                    df = pd.read_parquet(path, columns=["time", metric_name])
                    df = df.rename(columns={"time": "timestamp"})
            else:  # service
                path = os.path.join(
                    date_path, "apm", "service", f"service_{instance}_{date}.parquet"
                )
                if os.path.exists(path):
                    df = pd.read_parquet(path, columns=["time", metric_name])
                    df = df.rename(columns={"time": "timestamp"})
        elif metric_type == "infra_pod":
            path = os.path.join(
                date_path,
                "infra",
                "infra_pod",
                f"infra_pod_{metric_name}_{date}.parquet",
            )
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df = df[df["pod"] == instance]
                df = df[["time", metric_name]]
                df = df.rename(columns={"time": "timestamp"})
        elif metric_type == "infra_node":  # infra_node
            path = os.path.join(
                date_path,
                "infra",
                "infra_node",
                f"infra_node_{metric_name}_{date}.parquet",
            )
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df = df[df["kubernetes_node"] == instance]
                df = df[["time", metric_name]]
                df = df.rename(columns={"time": "timestamp"})
        # Handle TiDB specific metrics
        elif metric_type == "infra_tidb":
            path = os.path.join(
                date_path,
                "infra",
                "infra_tidb",
                f"infra_{init_metric_name}_{date}.parquet",
            )
            metric_name = init_metric_name.replace("tidb_", "")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df = df[["time", metric_name]]
                df = df.rename(columns={"time": "timestamp"})
        elif metric_type == "tidb_pd":
            path = os.path.join(
                date_path,
                "other",
                f"infra_{init_metric_name}_{date}.parquet",
            )
            metric_name = init_metric_name.replace("pd_", "")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df = df[["time", metric_name]]
                df = df.rename(columns={"time": "timestamp"})
        elif metric_type == "tidb_tikv":
            path = os.path.join(
                date_path,
                "other",
                f"infra_{init_metric_name}_{date}.parquet",
            )
            metric_name = init_metric_name.replace("tikv_", "")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                df = df[["time", metric_name]]
                df = df.rename(columns={"time": "timestamp"})

        if "df" in locals() and not df.empty:
            # Convert timestamp strings to datetime objects
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Filter by time range
            df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]
            if not df.empty:
                # Ensure DataFrame only contains timestamp and metric_value columns
                df = df[["timestamp", metric_name]].rename(
                    columns={metric_name: "metric_value"}
                )
                all_data.append(df)

    if not all_data:
        # print(
        #     f"No data found for metric {metric_name} and instance {instance} between {start_time} and {end_time}"
        # )
        return pd.DataFrame().to_numpy()

    result_df = pd.concat(all_data, ignore_index=True)
    # Sort by timestamp
    result_df = result_df.sort_values("timestamp")
    return result_df["metric_value"].to_numpy()


# def time_series_anomaly_detection_old(
#     start_time: Annotated[str, "start time of the anomaly"],
#     end_time: Annotated[str, "end time of the anomaly"],
#     metric_name: Annotated[str, "name of the metric"],
#     instance: Annotated[str, "name of the pod, service or node"],
# ) -> Tuple[np.ndarray, np.ndarray, dict]:
#     """
#     Perform time series anomaly detection using either 3-sigma rule or threshold method

#     Args:
#         start_time: Supports two formats:
#                    1. Beijing time format (e.g., '2024-09-12 20:08:19')
#                    2. UTC format (e.g., '2024-09-12T12:08:19Z')
#         end_time: Same formats as start_time
#         metric_name: Name of the metric
#         instance: Instance name (pod, service, or node), 'all' means all instances

#     Returns:
#         detection results
#     """
#     if "all" in instance.lower() and "all" in metric_name.lower():
#         all_nodes = [
#             "aiops-k8s-01",
#             "aiops-k8s-02",
#             "aiops-k8s-03",
#             "aiops-k8s-04",
#             "aiops-k8s-05",
#             "aiops-k8s-06",
#             "aiops-k8s-07",
#             "aiops-k8s-08",
#             "k8s-master1",
#             "k8s-master2",
#             "k8s-master3",
#         ]
#         all_service = [
#             "adservice",
#             "cartservice",
#             "currencyservice",
#             "productcatalogservice",
#             "checkoutservice",
#             "recommendationservice",
#             "shippingservice",
#             "emailservice",
#             "paymentservice",
#         ]
#         all_pod = []
#         for service in all_service:
#             for i in ["-0", "-1", "-2"]:
#                 all_pod.append(service + i)
#         results = []
#         for metric in apm_metric_names:
#             for service in all_service:
#                 results.append(
#                     time_series_anomaly_detection(start_time, end_time, metric, service)
#                 )
#         for metric in infra_node_metric_names:
#             for node in all_nodes:
#                 results.append(
#                     time_series_anomaly_detection(start_time, end_time, metric, node)
#                 )
#         for metric in infra_pod_metric_names:
#             for pod in all_pod:
#                 try:
#                     results.append(
#                         time_series_anomaly_detection(start_time, end_time, metric, pod)
#                     )
#                 except:
#                     print(f"Error in TSAD: {metric} {pod}")
#                     continue
#         return "\n".join([item for item in results if item != ""])
#     elif "all" in instance.lower() and _get_metric_type(metric_name) == "apm":
#         # all_service = [
#         #     "adservice",
#         #     "cartservice",
#         #     "currencyservice",
#         #     "productcatalogservice",
#         #     "checkoutservice",
#         #     "recommendationservice",
#         #     "shippingservice",
#         #     "emailservice",
#         #     "paymentservice",
#         # ]
#         results = []
#         for service in all_service:
#             results.append(
#                 time_series_anomaly_detection(
#                     start_time, end_time, metric_name, service
#                 )
#             )
#         return "\n".join([item for item in results if item != ""])
#     elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_pod":
#         # all_service = [
#         #     "adservice",
#         #     "cartservice",
#         #     "currencyservice",
#         #     "productcatalogservice",
#         #     "checkoutservice",
#         #     "recommendationservice",
#         #     "shippingservice",
#         #     "emailservice",
#         #     "paymentservice",
#         # ]
#         all_pod = []
#         for service in all_service:
#             for i in ["-0", "-1", "-2"]:
#                 all_pod.append(service + i)
#         results = []
#         for pod in all_pod:
#             results.append(
#                 time_series_anomaly_detection(start_time, end_time, metric_name, pod)
#             )
#         return "\n".join([item for item in results if item != ""])
#     elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_node":
#         # all_nodes = [
#         #     "aiops-k8s-01",
#         #     "aiops-k8s-02",
#         #     "aiops-k8s-03",
#         #     "aiops-k8s-04",
#         #     "aiops-k8s-05",
#         #     "aiops-k8s-06",
#         #     "aiops-k8s-07",
#         #     "aiops-k8s-08",
#         #     "k8s-master1",
#         #     "k8s-master2",
#         #     "k8s-master3",
#         # ]
#         results = []
#         for node in all_nodes:
#             results.append(
#                 time_series_anomaly_detection(start_time, end_time, metric_name, node)
#             )
#         return "\n".join([item for item in results if item != ""])
#     elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_tidb":
#         results = []
#         for pod in ["tidb-tidb"]:
#             results.append(
#                 time_series_anomaly_detection(start_time, end_time, metric_name, pod)
#             )
#         return "\n".join([item for item in results if item != ""])
#     elif "all" in instance.lower() and _get_metric_type(metric_name) == "tidb_pd":
#         results = []
#         for pod in ["tidb-pd"]:
#             results.append(
#                 time_series_anomaly_detection(start_time, end_time, metric_name, pod)
#             )
#         return "\n".join([item for item in results if item != ""])
#     elif "all" in instance.lower() and _get_metric_type(metric_name) == "tidb_tikv":
#         results = []
#         for pod in ["tidb-tikv"]:
#             results.append(
#                 time_series_anomaly_detection(start_time, end_time, metric_name, pod)
#             )
#         return "\n".join([item for item in results if item != ""])
#     else:
#         pass

#     history_minutes = 30
#     # Get anomaly period data
#     anomaly_values = get_metric_values_offline(
#         start_time=start_time,
#         end_time=end_time,
#         metric_name=metric_name,
#         instance=instance,
#     )

#     # For threshold-based detection, we don't need baseline data
#     if metric_name in THRESHOLD_METRICS:
#         baseline_values = np.array([])
#         anomaly_indices, stats = _detect_anomalies_threshold(
#             anomaly_values, metric_name
#         )
#     else:
#         # Parse and convert start time to UTC
#         start_utc = _parse_and_convert_time(start_time)
#         baseline_end = start_utc
#         baseline_start = baseline_end - timedelta(minutes=history_minutes)

#         # Convert baseline times back to Beijing time
#         beijing_tz = pytz.timezone("Asia/Shanghai")
#         baseline_start_beijing = baseline_start.astimezone(beijing_tz).strftime(
#             "%Y-%m-%d %H:%M:%S"
#         )
#         baseline_end_beijing = baseline_end.astimezone(beijing_tz).strftime(
#             "%Y-%m-%d %H:%M:%S"
#         )

#         # Get baseline period data
#         baseline_values = get_metric_values_offline(
#             start_time=baseline_start_beijing,
#             end_time=baseline_end_beijing,
#             metric_name=metric_name,
#             instance=instance,
#         )

#         # Use 3-sigma method
#         anomaly_indices, stats = _detect_anomalies_sigma(
#             anomaly_values, baseline_values
#         )

#     # Add detection method to stats
#     stats["detection_method"] = (
#         "threshold" if metric_name in THRESHOLD_METRICS else "3-sigma"
#     )

#     # Generate anomaly description
#     stats["description"] = _generate_anomaly_description(
#         metric_name=metric_name,
#         stats=stats,
#         values=anomaly_values,
#         baseline_values=baseline_values,
#         instance=instance,
#     )

#     return stats["description"]


def time_series_anomaly_detection(
    start_time: Annotated[str, "start time of the anomaly"],
    end_time: Annotated[str, "end time of the anomaly"],
    metric_name: Annotated[str, "name of the metric"],
    instance: Annotated[str, "name of the pod, service or node"],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Perform time series anomaly detection using either 3-sigma rule or threshold method

    Args:
        start_time: Supports two formats:
                   1. Beijing time format (e.g., '2024-09-12 20:08:19')
                   2. UTC format (e.g., '2024-09-12T12:08:19Z')
        end_time: Same formats as start_time
        metric_name: Name of the metric
        instance: Instance name (pod, service, or node), 'all' means all instances

    Returns:
        detection results
    """
    all_nodes = [
        "aiops-k8s-01",
        "aiops-k8s-02",
        "aiops-k8s-03",
        "aiops-k8s-04",
        "aiops-k8s-05",
        "aiops-k8s-06",
        "aiops-k8s-07",
        "aiops-k8s-08",
        "k8s-master1",
        "k8s-master2",
        "k8s-master3",
    ]
    all_service = [
        "adservice",
        "cartservice",
        "currencyservice",
        "productcatalogservice",
        "checkoutservice",
        "recommendationservice",
        "shippingservice",
        "emailservice",
        "paymentservice",
        "frontend",
        "paymentservice",
        "redis-cart",
    ]

    tidb_pods = ["tidb-tidb", "tidb-pd", "tidb-tikv"]

    all_pod = []
    for service in all_service:
        for i in ["-0", "-1", "-2"]:
            all_pod.append(service + i)

    if "all" in instance.lower() and "all" in metric_name.lower():
        # all_nodes = [
        #     "aiops-k8s-01",
        #     "aiops-k8s-02",
        #     "aiops-k8s-03",
        #     "aiops-k8s-04",
        #     "aiops-k8s-05",
        #     "aiops-k8s-06",
        #     "aiops-k8s-07",
        #     "aiops-k8s-08",
        #     "k8s-master1",
        #     "k8s-master2",
        #     "k8s-master3",
        # ]
        # all_service = [
        #     "adservice",
        #     "cartservice",
        #     "currencyservice",
        #     "productcatalogservice",
        #     "checkoutservice",
        #     "recommendationservice",
        #     "shippingservice",
        #     "emailservice",
        #     "paymentservice",
        # ]
        # all_pod = []
        # for service in all_service:
        #     for i in ["-0", "-1", "-2"]:
        #         all_pod.append(service + i)
        results = []
        for metric in apm_metric_names:
            for service in all_service:
                results.append(
                    time_series_anomaly_detection(start_time, end_time, metric, service)
                )
        for metric in infra_node_metric_names:
            for node in all_nodes:
                results.append(
                    time_series_anomaly_detection(start_time, end_time, metric, node)
                )
        for metric in infra_pod_metric_names:
            for pod in all_pod:
                try:
                    results.append(
                        time_series_anomaly_detection(start_time, end_time, metric, pod)
                    )
                except Exception as e:
                    print(f"Error in POD TSAD: {metric} {pod}. Exception: {e}")
                    continue
        for metric in infra_tidb_metric_names:
            try:
                results.append(
                    time_series_anomaly_detection(
                        start_time, end_time, metric, tidb_pods[0]
                    )
                )
            except Exception as e:
                print(f"Error in TIDB TSAD: {metric} {tidb_pods[0]}. Exception: {e}")
                continue
        for metric in tidb_pd_metric_names:
            try:
                results.append(
                    time_series_anomaly_detection(
                        start_time, end_time, metric, tidb_pods[1]
                    )
                )
            except Exception as e:
                print(f"Error in TIDB PD TSAD: {metric} {tidb_pods[1]}. Exception: {e}")
                continue
        for metric in tidb_tikv_metric_names:
            try:
                results.append(
                    time_series_anomaly_detection(
                        start_time, end_time, metric, tidb_pods[2]
                    )
                )
            except Exception as e:
                print(
                    f"Error in TIDB TiKV TSAD: {metric} {tidb_pods[2]}. Exception: {e}"
                )
                continue
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "apm":
        # all_service = [
        #     "adservice",
        #     "cartservice",
        #     "currencyservice",
        #     "productcatalogservice",
        #     "checkoutservice",
        #     "recommendationservice",
        #     "shippingservice",
        #     "emailservice",
        #     "paymentservice",
        # ]
        results = []
        for service in all_service:
            results.append(
                time_series_anomaly_detection(
                    start_time, end_time, metric_name, service
                )
            )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_pod":
        # all_service = [
        #     "adservice",
        #     "cartservice",
        #     "currencyservice",
        #     "productcatalogservice",
        #     "checkoutservice",
        #     "recommendationservice",
        #     "shippingservice",
        #     "emailservice",
        #     "paymentservice",
        # ]
        # all_pod = []
        # for service in all_service:
        #     for i in ["-0", "-1", "-2"]:
        #         all_pod.append(service + i)
        results = []
        for pod in all_pod:
            results.append(
                time_series_anomaly_detection(start_time, end_time, metric_name, pod)
            )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_node":
        # all_nodes = [
        #     "aiops-k8s-01",
        #     "aiops-k8s-02",
        #     "aiops-k8s-03",
        #     "aiops-k8s-04",
        #     "aiops-k8s-05",
        #     "aiops-k8s-06",
        #     "aiops-k8s-07",
        #     "aiops-k8s-08",
        #     "k8s-master1",
        #     "k8s-master2",
        #     "k8s-master3",
        # ]
        results = []
        for node in all_nodes:
            results.append(
                time_series_anomaly_detection(start_time, end_time, metric_name, node)
            )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_tidb":
        results = []
        results.append(
            time_series_anomaly_detection(
                start_time, end_time, metric_name, pod=tidb_pods[0]
            )
        )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "tidb_pd":
        results = []
        results.append(
            time_series_anomaly_detection(
                start_time, end_time, metric_name, pod=tidb_pods[1]
            )
        )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "tidb_tikv":
        results = []
        results.append(
            time_series_anomaly_detection(
                start_time, end_time, metric_name, pod=tidb_pods[2]
            )
        )
        return "\n".join([item for item in results if item != ""])
    else:
        pass

    history_minutes = WINDOW_SIZE - 1
    # Get anomaly period data
    anomaly_values = get_metric_values_offline(
        start_time=start_time,
        end_time=end_time,
        metric_name=metric_name,
        instance=instance,
    )
    # print(anomaly_values)
    # For threshold-based detection, we don't need baseline data
    if metric_name in THRESHOLD_METRICS:
        baseline_values = np.array([])
        anomaly_indices, stats = _detect_anomalies_threshold(
            anomaly_values, metric_name
        )
    elif metric_name in ML_METRICS:
        # Parse and convert start time to UTC
        start_utc = _parse_and_convert_time(start_time)
        baseline_end = start_utc
        baseline_start = baseline_end - timedelta(minutes=history_minutes)

        # Convert baseline times back to Beijing time
        beijing_tz = pytz.timezone("Asia/Shanghai")
        baseline_start_beijing = baseline_start.astimezone(beijing_tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        baseline_end_beijing = baseline_end.astimezone(beijing_tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Get baseline period data
        baseline_values = get_metric_values_offline(
            start_time=baseline_start_beijing,
            end_time=baseline_end_beijing,
            metric_name=metric_name,
            instance=instance,
        )

        # Use 3-sigma method
        # anomaly_indices, stats = _detect_anomalies_sigma(
        #     anomaly_values, baseline_values
        # )

        # Use FCVAE method
        anomaly_indices, stats = _detect_anomalies_fcvae(
            anomaly_values, baseline_values
        )
    else:  # 3-sigma
        # Parse and convert start time to UTC
        start_utc = _parse_and_convert_time(start_time)
        baseline_end = start_utc
        baseline_start = baseline_end - timedelta(minutes=history_minutes)

        # Convert baseline times back to Beijing time
        beijing_tz = pytz.timezone("Asia/Shanghai")
        baseline_start_beijing = baseline_start.astimezone(beijing_tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        baseline_end_beijing = baseline_end.astimezone(beijing_tz).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Get baseline period data
        baseline_values = get_metric_values_offline(
            start_time=baseline_start_beijing,
            end_time=baseline_end_beijing,
            metric_name=metric_name,
            instance=instance,
        )

        # Use 3-sigma method
        anomaly_indices, stats = _detect_anomalies_sigma(
            anomaly_values, baseline_values
        )

    # Add detection method to stats
    stats["detection_method"] = (
        "threshold" if metric_name in THRESHOLD_METRICS else "FCVAE"
    )

    # Generate anomaly description
    stats["description"] = _generate_anomaly_description(
        metric_name=metric_name,
        stats=stats,
        values=anomaly_values,
        baseline_values=baseline_values,
        instance=instance,
    )

    return stats["description"]


def _detect_anomalies_fcvae(
    values: np.ndarray, baseline_values: np.ndarray
) -> Tuple[np.ndarray, dict]:
    """
    使用 FCVAE 进行滑动窗口检测：每个窗口仅判定“最后一个点”是否为异常。

    流程：
    - 合并 baseline 与待检测序列，构造长度为 W 的滑动窗口（W = len(baseline)+1）。
    - 通过 FCVAE 输出每个窗口的重建分布，对最后一个点计算高斯对数似然（log-prob）。
    - 使用仅由“基线窗口”的最后点 log-prob 分布确定阈值（如 5% 分位）。
    - 对位于检测区间的窗口最后点，若 log-prob < 阈值，则标记该点为异常。

    返回：
      - anomaly_indices: 相对于 values 的索引（基于检测区间）
      - stats: 统计信息（包含阈值、比例、以及为兼容描述逻辑而提供的 mean/std 等）
    """
    # 基础健壮性
    values = np.asarray(values, dtype=np.float32)
    baseline_values = np.asarray(baseline_values, dtype=np.float32)

    if values.size == 0:
        return np.array([], dtype=int), {
            "anomaly_indices": np.array([], dtype=int),
            "anomaly_percentage": 0.0,
            "prob_threshold": None,
            "recon_log_prob": np.array([], dtype=float),
            # 兼容描述逻辑
            "mean": (
                float(np.mean(baseline_values)) if baseline_values.size > 0 else 0.0
            ),
            "std": float(np.std(baseline_values)) if baseline_values.size > 0 else 0.0,
            "upper_bound": None,
            "lower_bound": None,
        }

    # 窗口大小 W：history minutes = W - 1
    window_size = WINDOW_SIZE

    combined = np.concatenate([baseline_values, values], axis=0).astype(np.float32)

    # 对 combined 进行滑动平均，保持整个长度不变
    # 使用滑动窗口大小为 5，可以根据需要调整
    moving_avg_window = 5
    if combined.size >= moving_avg_window:
        # 使用卷积进行滑动平均
        kernel = np.ones(moving_avg_window) / moving_avg_window
        # 使用 edge 填充保持长度不变（复制边界值），再做有效卷积
        pad_left = moving_avg_window // 2
        pad_right = moving_avg_window - 1 - pad_left
        padded = np.pad(combined, (pad_left, pad_right), mode="edge")
        combined = np.convolve(padded, kernel, mode="valid")

    total_len = combined.size
    if total_len < window_size:
        # 样本不足，直接返回无异常
        return np.array([], dtype=int), {
            "anomaly_indices": np.array([], dtype=int),
            "anomaly_percentage": 0.0,
            "prob_threshold": None,
            "recon_log_prob": np.array([], dtype=float),
            # 兼容描述逻辑
            "mean": (
                float(np.mean(baseline_values)) if baseline_values.size > 0 else 0.0
            ),
            "std": float(np.std(baseline_values)) if baseline_values.size > 0 else 0.0,
            "upper_bound": None,
            "lower_bound": None,
        }

    # 构造滑动窗口 (num_windows, window_size)
    num_windows = total_len - window_size + 1
    X = np.lib.stride_tricks.sliding_window_view(combined, window_shape=window_size)
    X = X.astype(np.float32)
    # 对 X 的每个窗口分别进行 z-score 标准化
    # 形状: X -> (num_windows, window_size)
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    # 数值稳定：避免除零
    std = np.where(std < 1e-6, 1.0, std)
    X = (X - mu) / std

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FCVAE 模型构建（尽量与已训练权重结构相容；若加载失败，使用随机权重也可运行）
    # 经验参数：latent_dim=8, kernel_size=min(24, window_size//2), stride=max(1, kernel_size//3), condition_emb_dim=64
    kernel_size = 24
    stride = 8
    condition_emb_dim = 64
    model = FCVAE(
        input_dim=window_size,
        latent_dim=8,
        kernel_size=kernel_size,
        stride=stride,
        condition_emb_dim=condition_emb_dim,
        dropout_rate=0.05,
    ).to(device)

    # 尝试加载权重（严格性放宽）
    ckpt_path_candidates = [
        os.path.join(os.path.dirname(__file__), "fcvae_model_final.pth"),
        os.path.join(os.path.dirname(__file__), "..", "tools", "fcvae_model_final.pth"),
        os.path.join(os.getcwd(), "tools", "fcvae_model_final.pth"),
        os.path.join(os.getcwd(), "fcvae_model_final.pth"),
    ]
    for ckpt in ckpt_path_candidates:
        try:
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state, strict=False)
                break
        except Exception:
            pass

    model.eval()

    # 前向计算重建分布，得到最后一点的 log-prob
    with torch.no_grad():
        tx = torch.from_numpy(X).to(device)
        recon_mu, recon_var, _, _, _ = model(tx)
        var = recon_var.clamp_min(1e-8)
        last_idx = -1
        x_last = tx[:, last_idx]
        mu_last = recon_mu[:, last_idx]
        var_last = var[:, last_idx]
        # log N(x|mu,var) = -0.5 * [ log(2πvar) + (x-mu)^2/var ]
        log_prob = -0.5 * (
            torch.log(2 * torch.pi * var_last) + (x_last - mu_last) ** 2 / var_last
        )
        log_prob = log_prob.detach().cpu().numpy()
        print(log_prob.mean(), log_prob.std())
        # 可视化：每个窗口最后一个点的原始值与重建均值（标准化尺度）
        orig_last = x_last.detach().cpu().numpy()
        recon_last = mu_last.detach().cpu().numpy()
        # try:
        #     plt.figure(figsize=(10, 4))
        #     plt.plot(orig_last, label="original(last)")
        #     plt.plot(recon_last, label="recon_mu(last)")
        #     plt.xlabel("window index")
        #     plt.ylabel("value (z-scored per window)")
        #     plt.title("Reconstruction vs Original (last point per window)")
        #     plt.legend()
        #     out_path = os.path.join(os.path.dirname(__file__), "reconstruction_last.png")
        #     plt.tight_layout()
        #     plt.savefig(out_path, dpi=150)
        #     plt.close()
        #     print(f"Saved plot to {out_path}")

        #     # 将最后一点从 z-score 还原回原始尺度：x = x_norm * std + mu
        #     std_np = np.squeeze(std, axis=1)
        #     mu_np = np.squeeze(mu, axis=1)
        #     orig_last_denorm = orig_last * std_np + mu_np
        #     recon_last_denorm = recon_last * std_np + mu_np

        #     plt.figure(figsize=(10, 4))
        #     plt.plot(orig_last_denorm, label="original(last, denorm)")
        #     plt.plot(recon_last_denorm, label="recon_mu(last, denorm)")
        #     plt.xlabel("window index")
        #     plt.ylabel("value (original scale)")
        #     plt.title("Reconstruction vs Original (denormalized, last point per window)")
        #     plt.legend()
        #     out_path_denorm = os.path.join(os.path.dirname(__file__), "reconstruction_last_denorm.png")
        #     plt.tight_layout()
        #     plt.savefig(out_path_denorm, dpi=150)
        #     plt.close()
        #     print(f"Saved plot to {out_path_denorm}")
        # except Exception as _:
        #     pass

    prob_threshold = -5
    detect_indices = np.arange(len(log_prob))

    anomaly_point_global_indices = []
    for i in detect_indices:
        if log_prob[i] < prob_threshold:
            anomaly_point_global_indices.append(i)
    anomaly_indices = np.array(sorted(set(anomaly_point_global_indices)), dtype=int)

    stats = {
        "prob_threshold": prob_threshold,
        "recon_log_prob": log_prob,
        "anomaly_indices": anomaly_indices,
        "anomaly_percentage": (anomaly_indices.size / max(1, values.size)),
    }

    return anomaly_indices, stats


def get_trace_values_offline(
    start_time: Annotated[str, "Start time"],
    end_time: Annotated[str, "End time"],
) -> pd.DataFrame:
    """
    Extract average duration information from trace parquet files

    Args:
        start_time: Supports two formats:
                   1. Beijing time format (e.g., '2024-09-12 20:08:19')
                   2. UTC format (e.g., '2024-09-12T12:08:19Z')
        end_time: Same formats as start_time

    Returns:
        DataFrame containing average duration and pod information for each operationName
    """
    # Parse and convert times to UTC, then convert to Beijing time for file processing
    start_utc = _parse_and_convert_time(start_time)
    end_utc = _parse_and_convert_time(end_time)

    beijing_tz = pytz.timezone("Asia/Shanghai")
    start_dt = start_utc.astimezone(beijing_tz)
    end_dt = end_utc.astimezone(beijing_tz)

    # Get date range
    dates = []
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    all_data = []
    # base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )

    for date in dates:
        # date_folder = f"trace-parquet_{date}"
        date_folder = f"{date.replace('-', '')}/trace-parquet"
        date_path = os.path.join(base_path, date_folder)

        if not os.path.exists(date_path):
            continue

        # Determine hour range for current date (using Beijing time)
        if date == dates[0]:  # First day
            start_hour = start_dt.hour
        else:
            start_hour = 0

        if date == dates[-1]:  # Last day
            end_hour = end_dt.hour
        else:
            end_hour = 23

        # Process each hour file
        for hour in range(start_hour, end_hour + 1):
            file_name = f"trace_jaeger-span_{date}_{hour:02d}-59-00.parquet"
            file_path = os.path.join(date_path, file_name)

            if not os.path.exists(file_path):
                continue

            # Read parquet file
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Trace file corrupt, error in reading. Exception: {e}")
                continue

            # Convert timestamp to Beijing time
            df["timestamp"] = pd.to_datetime(
                df["startTimeMillis"], unit="ms", utc=True
            ).dt.tz_convert("Asia/Shanghai")

            # Filter by time range
            df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

            if not df.empty:
                # Process operationName
                def clean_operation_name(name):
                    # Remove possible prefixes
                    name = name.lower()
                    if "grpc." in name:
                        name = name.split("grpc.")[-1]
                    if "/hipstershop." in name:
                        name = name.split("/hipstershop.")[-1]
                    if "hipstershop." in name:
                        name = name.split("hipstershop.")[-1]
                    # Keep only service name part (remove method name)
                    if "/" in name:
                        name = name.split("/")[0]
                    return name

                df["operationName"] = df["operationName"].apply(clean_operation_name)

                # Keep only operations containing "service"
                df = df[df["operationName"].str.contains("service", case=False)]

                # Extract pod information
                def extract_pod_name(process):
                    if isinstance(process, dict) and "tags" in process:
                        for tag in process["tags"]:
                            if isinstance(tag, dict) and tag.get("key") == "name":
                                return tag.get("value", "")
                    return ""

                df["pod"] = df["process"].apply(extract_pod_name)

                # Select needed columns
                df = df[["timestamp", "operationName", "duration", "pod"]]
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    # Merge all data
    result_df = pd.concat(all_data, ignore_index=True)

    # Calculate average duration for each operationName
    summary_df = (
        result_df.groupby(["operationName", "pod"])["duration"]
        .agg([("avg_duration", "mean"), ("count", "count")])
        .reset_index()
    )

    return summary_df


def trace_anomaly_detection(
    start_time: Annotated[str, "Start time"],
    end_time: Annotated[str, "End time"],
) -> str:
    """
    Detect anomalies in trace calls by comparing current period with historical baseline

    Args:
        start_time: Supports two formats:
                   1. Beijing time format (e.g., '2024-09-12 20:08:19')
                   2. UTC format (e.g., '2024-09-12T12:08:19Z')
        end_time: Same formats as start_time
    Returns:
        Text description of detected anomalies
    """
    history_minutes = 30

    # Parse and convert start time to UTC
    start_utc = _parse_and_convert_time(start_time)
    end_utc = _parse_and_convert_time(end_time)

    # Calculate baseline period
    baseline_end = start_utc
    baseline_start = baseline_end - timedelta(minutes=history_minutes)

    # Convert baseline times to Beijing format for consistency
    beijing_tz = pytz.timezone("Asia/Shanghai")
    baseline_start_beijing = baseline_start.astimezone(beijing_tz).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    baseline_end_beijing = baseline_end.astimezone(beijing_tz).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Get current period trace data
    current_df = get_trace_values_offline(start_time, end_time)
    if current_df.empty:
        return "No trace data found"

    # Get baseline period trace data
    baseline_df = get_trace_values_offline(baseline_start_beijing, baseline_end_beijing)
    if baseline_df.empty:
        return "No baseline trace data found"

    # Calculate total request counts
    current_total_count = current_df["count"].sum()
    baseline_total_count = baseline_df["count"].sum()

    # Analyze anomalies
    duration_anomalies = []
    count_anomalies = []

    for _, current_row in current_df.iterrows():
        service = current_row["operationName"]
        pod = current_row["pod"]
        current_duration = current_row["avg_duration"]
        # Calculate current request proportion
        current_ratio = (
            current_row["count"] / current_total_count * 100
        )  # Convert to percentage

        # Find corresponding baseline data
        baseline_data = baseline_df[
            (baseline_df["operationName"] == service) & (baseline_df["pod"] == pod)
        ]

        if not baseline_data.empty:
            baseline_duration = baseline_data.iloc[0]["avg_duration"]
            # Calculate baseline request proportion
            baseline_ratio = (
                baseline_data.iloc[0]["count"] / baseline_total_count * 100
            )  # Convert to percentage

            # Calculate percentage changes
            duration_change_pct = (
                (current_duration - baseline_duration) / baseline_duration
            ) * 100
            ratio_change_pct = (
                ((current_ratio - baseline_ratio) / baseline_ratio) * 100
                if baseline_ratio > 0
                else float("inf")
            )

            # Detect latency anomalies
            if abs(duration_change_pct) > 100:  # Latency change exceeds 100%
                change_type = "increased" if duration_change_pct > 0 else "decreased"
                duration_anomalies.append(
                    f"Service {service} (Caller: {pod}): latency has {change_type} by {abs(duration_change_pct):.1f}%"
                )

            # Detect request proportion anomalies
            if abs(ratio_change_pct) > 100:  # Request proportion change exceeds 100%
                change_type = "increased" if ratio_change_pct > 0 else "decreased"
                count_anomalies.append(
                    f"Service {service} (Caller: {pod}): request proportion has {change_type} by {abs(ratio_change_pct):.1f}% "
                    + f"(from {baseline_ratio:.1f}% to {current_ratio:.1f}% of total requests)"
                )

    # Generate final report
    if not duration_anomalies and not count_anomalies:
        return "No anomalies detected during the specified time period"

    report_parts = []

    if duration_anomalies:
        report_parts.append("Latency Anomalies:\n" + "\n".join(duration_anomalies))

    if count_anomalies:
        if report_parts:  # Add empty line if there are latency anomalies
            report_parts.append("")
        report_parts.append(
            "Request Proportion Anomalies:\n" + "\n".join(count_anomalies)
        )

    return "\n".join(report_parts)


def get_logs_offline(
    start_time: Annotated[str, "Start time"],
    end_time: Annotated[str, "End time"],
) -> pd.DataFrame:
    """
    Extract log data from log parquet files for hipstershop namespace

    Args:
        start_time: Supports two formats:
                   1. Beijing time format (e.g., '2024-09-12 20:08:19')
                   2. UTC format (e.g., '2024-09-12T12:08:19Z')
        end_time: Same formats as start_time

    Returns:
        DataFrame containing pod name, timestamp, and log message information for hipstershop namespace
    """
    # Define valid pods
    all_service = [
        "adservice",
        "cartservice",
        "currencyservice",
        "productcatalogservice",
        "checkoutservice",
        "recommendationservice",
        "shippingservice",
        "emailservice",
        "paymentservice",
    ]
    all_pod = []
    for service in all_service:
        for i in ["-0", "-1", "-2"]:
            all_pod.append(service + i)

    # Parse and convert times to UTC, then convert to Beijing time for file processing
    start_utc = _parse_and_convert_time(start_time)
    end_utc = _parse_and_convert_time(end_time)

    beijing_tz = pytz.timezone("Asia/Shanghai")
    start_dt = start_utc.astimezone(beijing_tz)
    end_dt = end_utc.astimezone(beijing_tz)

    # Get date range
    dates = []
    current_date = start_dt.date()
    while current_date <= end_dt.date():
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    all_data = []
    # base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
    )

    for date in dates:
        # date_folder = f"log-parquet_{date}"
        date_folder = f"{date.replace('-', '')}/log-parquet"
        date_path = os.path.join(base_path, date_folder)

        if not os.path.exists(date_path):
            continue

        # Determine hour range for current date (using Beijing time)
        if date == dates[0]:  # First day
            start_hour = start_dt.hour
        else:
            start_hour = 0

        if date == dates[-1]:  # Last day
            end_hour = end_dt.hour
        else:
            end_hour = 23

        # Process each hour file
        for hour in range(start_hour, end_hour + 1):
            file_name = f"log_filebeat-server_{date}_{hour:02d}-59-00.parquet"
            file_path = os.path.join(date_path, file_name)

            if not os.path.exists(file_path):
                continue

            # Read parquet file
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"Log file corrupt, error in reading. Exception: {e}")
                continue

            # Filter for hipstershop namespace
            df = df[df["k8_namespace"] == "hipstershop"]

            if df.empty:
                continue

            # Extract required fields
            df["pod"] = df["k8_pod"]

            # Filter for specific pods
            df = df[df["pod"].isin(all_pod)]

            if df.empty:
                continue

            # Convert timestamp to UTC
            df["timestamp"] = pd.to_datetime(df["@timestamp"])

            # Extract message from the message JSON string
            def extract_message(msg_str):
                if not isinstance(msg_str, str):
                    return str(msg_str)

                # Check if the string starts and ends with curly braces (potential JSON)
                if msg_str.strip().startswith("{") and msg_str.strip().endswith("}"):
                    try:
                        import json

                        msg_json = json.loads(msg_str)
                        return msg_json.get("message", msg_str)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        return msg_str
                return msg_str

            df["message"] = df["message"].apply(extract_message)

            # Filter by time range
            df = df[(df["timestamp"] >= start_utc) & (df["timestamp"] <= end_utc)]

            if not df.empty:
                # Select only needed columns
                df = df[["pod", "timestamp", "message"]]
                all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    # Merge all data
    result_df = pd.concat(all_data, ignore_index=True)

    # Sort by timestamp
    result_df = result_df.sort_values("timestamp")

    return result_df


# def log_anomaly_detection_old(
#     start_time: Annotated[str, "Start time"],
#     end_time: Annotated[str, "End time"],
# ) -> str:
#     """
#     Detect error logs by searching for specific keywords in log messages

#     Args:
#         start_time: Supports two formats:
#                    1. Beijing time format (e.g., '2024-09-12 20:08:19')
#                    2. UTC format (e.g., '2024-09-12T12:08:19Z')
#         end_time: Same formats as start_time

#     Returns:
#         Text description of detected error logs, with duplicate messages removed
#     """
#     # Get logs for the specified time period
#     logs_df = get_logs_offline(start_time, end_time)

#     if logs_df.empty:
#         return "No logs found for the specified time period"

#     # Define error keywords (case insensitive)
#     error_keywords = ["error", "fail", "exception"]

#     # Create regex pattern for error keywords
#     pattern = "|".join(error_keywords)

#     # Filter logs containing error keywords
#     error_logs = logs_df[
#         logs_df["message"].str.lower().str.contains(pattern, case=False, na=False)
#     ]

#     if error_logs.empty:
#         return "No error logs detected during the specified time period"

#     # Group errors by pod, using sets to remove duplicates
#     pod_errors = {}
#     for _, row in error_logs.iterrows():
#         pod = row["pod"]
#         timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
#         message = row["message"]

#         if pod not in pod_errors:
#             pod_errors[pod] = (
#                 {}
#             )  # Using dict to store unique messages with their first occurrence time

#         if message not in pod_errors[pod]:
#             pod_errors[pod][message] = timestamp

#     # Generate report
#     report_parts = ["Error Logs Detected:"]

#     for pod, messages in pod_errors.items():
#         report_parts.append(f"\nPod: {pod}")
#         report_parts.append("-" * (len(pod) + 5))
#         for msg, timestamp in messages.items():
#             report_parts.append(f"[{timestamp}] {msg}")

#     return "\n".join(report_parts)


def log_anomaly_detection(
    start_time: Annotated[str, "Start time"],
    end_time: Annotated[str, "End time"],
) -> str:
    """
    Detect error logs by searching for specific keywords in log messages

    Args:
        start_time: Supports two formats:
                   1. Beijing time format (e.g., '2024-09-12 20:08:19')
                   2. UTC format (e.g., '2024-09-12T12:08:19Z')
        end_time: Same formats as start_time

    Returns:
        Text description of detected error logs, with duplicate messages removed
    """
    # Get logs for the specified time period
    logs_df = get_logs_offline(start_time, end_time)

    if logs_df.empty:
        return "No logs found for the specified time period"

    # Define error keywords (case insensitive)
    # error_keywords = ["error", "fail", "exception"]
    error_keywords = [
        "error",
        "failed",
        "failure",
        "exception",
        "timeout",
        "crash",
        "disconnect",
        "refused",
        "rejected",
        "aborted",
        "unavailable",
        "misbehaving",
        "timeout",
        "retry",
        "conflict",
        "deadlock",
        "rollback",
        "abort",
        "stall",
    ]

    # Create regex pattern for error keywords
    pattern = "|".join(error_keywords)

    # Filter logs containing error keywords
    error_logs = logs_df[
        logs_df["message"].str.lower().str.contains(pattern, case=False, na=False)
    ]

    if error_logs.empty:
        return "No error logs detected during the specified time period"

    # Group errors by pod, using sets to remove duplicates
    pod_errors = {}
    for _, row in error_logs.iterrows():
        pod = row["pod"]
        timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        message = row["message"]

        if pod not in pod_errors:
            pod_errors[pod] = (
                {}
            )  # Using dict to store unique messages with their first occurrence time

        if message not in pod_errors[pod]:
            pod_errors[pod][message] = timestamp

    # 构造提示词并调用大模型生成关键词与总结
    # 为避免提示超长，对每个 pod 的样例条目和消息长度进行限制
    max_examples_per_pod = 50
    max_message_chars = 300

    def truncate_message(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if len(text) > max_message_chars:
            return text[: max_message_chars - 3] + "..."
        return text

    lines = []
    for pod, messages in pod_errors.items():
        lines.append(f"Pod: {pod}")
        count = 0
        for msg, timestamp in messages.items():
            if count >= max_examples_per_pod:
                break
            lines.append(f"[{timestamp}] {truncate_message(msg)}")
            count += 1
        lines.append("")

    logs_block = "\n".join(lines)

    prompt = (
        "You are an expert SRE log analysis assistant. Given error logs grouped by pod, "
        "please extract concise error keywords and write a short summary.\n\n"
        "Requirements:\n"
        "- Return English output.\n"
        "- For each Pod, provide 3-8 high-signal keywords/phrases capturing the core error patterns.\n"
        "- Then write an overall brief summary (3-6 sentences): suspected cause, scope/impacted services, and rough timeline cues if any.\n"
        "- Be specific, avoid redundant wording.\n\n"
        "Logs (deduplicated, time ascending):\n"
        f"{logs_block}\n\n"
        "Output format strictly as below (no extra text):\n"
        "Pod: <pod-name>\n"
        "Keywords: <k1>, <k2>, <k3>\n\n"
        "... (repeat for all pods)\n\n"
        "Summary: <overall summary in English>"
    )

    try:
        response = chat(prompt)
    except Exception as e:
        # 兜底返回本地汇总
        fallback = ["Error Logs Detected:"] + lines
        fallback.append("")
        fallback.append(f"LLM 调用失败: {str(e)}")
        return "\n".join(fallback)

    return response


def classifier(
    start_time: Annotated[str, "Start time of the anomaly"],
    end_time: Annotated[str, "End time of the anomaly"],
) -> str:
    """
    Use anomaly detection results and determine the main type of failure

    Args:
        start_time: Start time of the anomaly (e.g., '2025-05-07T11:10:44Z')
        end_time: End time of the anomlay

    Returns:
        Classification result text containing:
        1. Main failure type (possible types include):
        2. Reasoning explanation highlighting the key signals or information that led to this conclusion
    """
    result = time_series_anomaly_detection(start_time, end_time, "all", "all")
    prompt = f"""You are a cloud operations assistant specializing in analyzing anomaly detection results from Kubernetes clusters. Given anomaly detection result (The anomaly detection results may contain many false positives; you can filter and make judgments based on information such as the anomaly ratio.) — DETERMINE the main type of failure.

The following are some correspondences between fault phenomena and fault types:
NODE CPU USAGE RATE increases: Node CPU
NODE MEMORY USAGE increases: Node Memory
POD CPU USAGE RATE and POD PROCESSES increases: Pod CPU
POD PROCESSES increases: Pod Memory
High Node CPU and Higt Pod CPU: Jvm CPU
High RRT, Low Request, Low Response and Low Network Transmit Packets: Network Delay
High network transmit packets and high error ratio: Network Duplicate
High Error Ratio, High RRT, High Server Error Ratio, Low Request, Low Response and Low Network Transmit Packets: DNS Error
High Node CPU, Higt Pod CPU and High RTT: Jvm Lantency
High RTT, High ERROR_RATIO and Low Network Transmit Packets: Network Loss
High RRT, Low Request, Low Response: Network Corrupt
LOW Request, Low Response, Low Network Transmit Packets and High Pod Processes: Pod Kill
Multiple Metrics to Zero such and Multiple Metrics loss: Pod Failure

All fault Types:
- pod kill
- pod failure
- network delay
- network duplicate
- network loss
- network corrupt
- node cpu
- node memory
- pod cpu
- pod memory
- jvm cpu
- dns error
- jvm lantency

Anomlay detection result:
{result}

Just give me the main failure type (choose one from the list above) without any other output.

Example output:
Main failure type: pod memory"""
    response = chat(prompt)
    return response


def query_system_information() -> str:
    return """
The system information is as follows:

**All Nodes**:
['aiops-k8s-01', 'aiops-k8s-02', 'aiops-k8s-03', 'aiops-k8s-04', 'aiops-k8s-05', 'aiops-k8s-06','aiops-k8s-07', 'aiops-k8s-08', 'k8s-master1', 'k8s-master2', 'k8s-master3']

**All services**:
['adservice', 'cartservice', 'currencyservice', 'productcatalogservice', 'checkoutservice','recommendationservice', 'shippingservice','emailservice', 'paymentservice']
Each service has three pods, e.g. adservice has three pods adservice-0, adservice-1, adservice-2

**Relationship between Services**:
A->B means service A call service B.
frontend->adservice,frontend->productcatalogservice,frontend->currencyservice,frontend->recommendationservice,cartservice,checkoutservice->paymentservice->shippingservice->emailservice


**All metrics**:
apm_metric_names = ["client_error_ratio","error_ratio","request","response","rrt","server_error_ratio","timeout"]
infra_pod_metric_names = ["pod_cpu_usage", "pod_fs_reads_bytes", "pod_fs_writes_bytes", "pod_network_receive_bytes", "pod_network_receive_packets", "pod_network_transmit_bytes", "pod_network_transmit_packets", "pod_processes"]
infra_node_metric_names = ["node_cpu_usage_rate", "node_filesystem_usage_rate", "node_memory_usage_rate", "node_network_receive_packets_total", "node_network_transmit_packets_total", "node_sockstat_TCP_inuse"]


**Kubernetes info**
NAME                            READY   STATUS      RESTARTS        AGE     IP              NODE           NOMINATED NODE   READINESS GATES
adservice-0                     1/1     Running     8 (4m14s ago)   7d5h    10.233.89.223   aiops-k8s-08   <none>           <none>
adservice-1                     1/1     Running     12 (14m ago)    7d5h    10.233.81.32    aiops-k8s-03   <none>           <none>
adservice-2                     1/1     Running     5 (63m ago)     7d5h    10.233.85.74    aiops-k8s-07   <none>           <none>
cartservice-0                   1/1     Running     2 (14h ago)     7d5h    10.233.81.73    aiops-k8s-03   <none>           <none>
cartservice-1                   1/1     Running     2 (14h ago)     7d5h    10.233.78.180   aiops-k8s-01   <none>           <none>
cartservice-2                   1/1     Running     2 (14h ago)     7d5h    10.233.77.46    aiops-k8s-04   <none>           <none>
checkoutservice-0               1/1     Running     1 (3d ago)      7d5h    10.233.85.52    aiops-k8s-07   <none>           <none>
checkoutservice-1               1/1     Running     1 (3d ago)      7d5h    10.233.81.234   aiops-k8s-03   <none>           <none>
checkoutservice-2               1/1     Running     1 (3d ago)      7d5h    10.233.77.2     aiops-k8s-04   <none>           <none>
currencyservice-0               1/1     Running     2 (7h3m ago)    2d22h   10.233.85.212   aiops-k8s-07   <none>           <none>
currencyservice-1               1/1     Running     2 (7h3m ago)    2d22h   10.233.81.6     aiops-k8s-03   <none>           <none>
currencyservice-2               1/1     Running     2 (7h3m ago)    2d22h   10.233.79.185   aiops-k8s-06   <none>           <none>
emailservice-0                  1/1     Running     4 (11h ago)     7d5h    10.233.85.75    aiops-k8s-07   <none>           <none>
emailservice-1                  1/1     Running     4 (11h ago)     7d5h    10.233.79.60    aiops-k8s-06   <none>           <none>
emailservice-2                  1/1     Running     4 (11h ago)     7d5h    10.233.78.139   aiops-k8s-01   <none>           <none>
example-ant-29107680-n6c8f      1/1     Running     0               15h     10.233.74.15    aiops-k8s-05   <none>           <none>
frontend-0                      1/1     Running     0               7d5h    10.233.85.77    aiops-k8s-07   <none>           <none>
frontend-1                      1/1     Running     0               7d5h    10.233.81.88    aiops-k8s-03   <none>           <none>
frontend-2                      1/1     Running     0               7d5h    10.233.74.117   aiops-k8s-05   <none>           <none>
paymentservice-0                1/1     Running     3 (19h ago)     7d5h    10.233.81.216   aiops-k8s-03   <none>           <none>
paymentservice-1                1/1     Running     3 (19h ago)     7d5h    10.233.89.213   aiops-k8s-08   <none>           <none>
paymentservice-2                1/1     Running     3 (19h ago)     7d5h    10.233.78.103   aiops-k8s-01   <none>           <none>
productcatalogservice-0         1/1     Running     1 (2d23h ago)   3d6h    10.233.74.105   aiops-k8s-05   <none>           <none>
productcatalogservice-1         1/1     Running     1 (2d23h ago)   3d6h    10.233.81.242   aiops-k8s-03   <none>           <none>
productcatalogservice-2         1/1     Running     1 (2d23h ago)   3d6h    10.233.85.21    aiops-k8s-07   <none>           <none>
recommendationservice-0         1/1     Running     1 (25h ago)     2d7h    10.233.85.42    aiops-k8s-07   <none>           <none>
recommendationservice-1         1/1     Running     1 (25h ago)     2d7h    10.233.81.146   aiops-k8s-03   <none>           <none>
recommendationservice-2         1/1     Running     1 (25h ago)     2d7h    10.233.89.86    aiops-k8s-08   <none>           <none>
redis-cart-0                    1/1     Running     0               7d5h    10.233.89.187   aiops-k8s-08   <none>           <none>
shippingservice-0               1/1     Running     0               2d19h   10.233.81.210   aiops-k8s-03   <none>           <none>
shippingservice-1               1/1     Running     0               2d19h   10.233.85.163   aiops-k8s-07   <none>           <none>
shippingservice-2               1/1     Running     0               2d19h   10.233.89.25    aiops-k8s-08   <none>           <none>
"""


# def query_system_information(query: Annotated[str, "query question"]) -> str:
#     """
#     Use RAG approach to retrieve information from system_info.txt

#     Args:
#         query: User's query question

#     Returns:
#         Retrieved relevant information
#     """
#     # Read system_info.txt file
#     base_path = os.path.dirname(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     )
#     system_info_path = os.path.join(base_path, "data", "system_info.txt")
#     if not os.path.exists(system_info_path):
#         return "system_info.txt file not found"

#     try:
#         with open(system_info_path, "r", encoding="utf-8") as f:
#             content = f.read()
#     except Exception as e:
#         return f"Failed to read system_info.txt: {str(e)}"

#     # Split text into paragraphs
#     paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

#     # Construct prompt
#     prompt = f"""As a system information retrieval assistant, please help me find the most relevant information from the system information below.

# Question: {query}

# System Information:
# {content}

# Please return only the relevant information without any explanation or comments. If no relevant information is found, return "No relevant information found"."""

#     # Use LLM for retrieval
#     response = chat(prompt)

#     return response


# result = trace_anomaly_detection("2025-05-06T00:00:00Z", "2025-05-06T00:10:00Z")
# print(result)

# result = log_anomaly_detection_llm("2025-05-06T00:00:00Z", "2025-05-07T00:10:00Z")
# print(result)
# result = time_series_anomaly_detection("2025-05-05T10:11:31Z", "2025-05-05T10:29:31Z", "node_cpu_usage_rate", "all")
# print(result)

if __name__ == "__main__":
    base_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Test code here if needed
    with open(os.path.join(base_path, "data", "label_test.json"), "rb") as f:
        labels = json.load(f)
        # labels = labels[:1]  # For testing, limit to first 50 entries

    for item in labels:
        print(item)
        start_time = item["start_time"]
        end_time = item["end_time"]
        time_result = time_series_anomaly_detection(start_time, end_time, "all", "all")
        log_result = log_anomaly_detection(start_time, end_time)
        trace_result = trace_anomaly_detection(start_time, end_time)
        item["log_result"] = log_result
        item["trace_result"] = trace_result
        item["time_series_result"] = time_result

    with open(
        os.path.join(base_path, "result", "anomaly_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
        # f.write(result)
