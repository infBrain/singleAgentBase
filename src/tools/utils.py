from typing import Annotated, List, Tuple
import os
import pandas as pd
from datetime import datetime, timedelta
import pytz
import re
import numpy as np 

# apm_metric_names = ["client_error","client_error_ratio","error","error_ratio","request","response","rrt","rrt_max","server_error","server_error_ratio","timeout"]
apm_metric_names = ["client_error_ratio","error_ratio","request","response","rrt","server_error_ratio","timeout"]
# infra_pod_metric_names = ["pod_cpu_usage", "pod_fs_reads_bytes", "pod_fs_writes_bytes", "pod_memory_working_set_bytes", "pod_network_receive_bytes", "pod_network_receive_packets", "pod_network_transmit_bytes", "pod_network_transmit_packets", "pod_processes"]
infra_pod_metric_names = ["pod_cpu_usage", "pod_fs_reads_bytes", "pod_fs_writes_bytes", "pod_network_receive_bytes", "pod_network_receive_packets", "pod_network_transmit_bytes", "pod_network_transmit_packets", "pod_processes"]
# infra_node_metric_names = ["node_cpu_usage_rate", "node_disk_read_bytes_total", "node_disk_read_time_seconds_total", "node_disk_write_time_seconds_total", "node_disk_written_bytes_total", "node_filesystem_free_bytes", "node_filesystem_size_bytes", "node_filesystem_usage_rate", "node_memory_MemAvailable_bytes", "node_memory_MemTotal_bytes", "node_memory_usage_rate", "node_network_receive_bytes_total", "node_network_receive_packets_total", "node_network_transmit_bytes_total", "node_network_transmit_packets_total", "node_sockstat_TCP_inuse"]
infra_node_metric_names = ["node_cpu_usage_rate", "node_filesystem_usage_rate", "node_memory_usage_rate", "node_network_receive_packets_total", "node_network_transmit_packets_total", "node_sockstat_TCP_inuse"]

infra_tidb_metric_names = ['tidb_block_cache_size','tidb_connection_count','tidb_cpu_usage','tidb_ddl_job_count','tidb_duration_95th','tidb_duration_99th','tidb_duration_avg','tidb_failed_query_ops','tidb_memory_usage','tidb_qps','tidb_server_is_up','tidb_slow_query','tidb_top_sql_cpu','tidb_transaction_retry','tidb_uptime']
tidb_pd_metric_names = ['pd_abnormal_region_count', 'pd_leader_count', 'pd_leader_primary', 'pd_learner_count', 'pd_region_count', 'pd_region_health', 'pd_storage_capacity', 'pd_storage_size', 'pd_storage_used_ratio', 'pd_store_down_count', 'pd_store_low_space_count', 'pd_store_slow_count', 'pd_store_unhealth_count', 'pd_store_up_count', 'pd_witness_count']
tidb_tikv_metric_names = ['tikv_available_size', 'tikv_capacity_size', 'tikv_cpu_usage', 'tikv_grpc_qps', 'tikv_io_util', 'tikv_memory_usage', 'tikv_qps', 'tikv_raft_apply_wait', 'tikv_raft_propose_wait', 'tikv_read_mbps', 'tikv_region_pending', 'tikv_rocksdb_write_stall', 'tikv_server_is_up', 'tikv_snapshot_apply_count', 'tikv_store_size', 'tikv_threadpool_readpool_cpu', 'tikv_write_wal_mbps']

# tidb_metric_names_total = infra_tidb_metric_names + infra_pd_metric_names + infra_tikv_metric_names

# Define metric detection methods and thresholds
THRESHOLD_METRICS = {
    # APM metrics (ratio metrics should use threshold)
    # 'client_error_ratio': 2,    # 10% error ratio threshold
    # 'error_ratio': 2,           # 10% error ratio threshold
    # 'server_error_ratio': 2,    # 10% error ratio threshold
    'rrt': 20000,                  # 1000ms response time threshold
    
    # Infrastructure Pod metrics
    # 'pod_cpu_usage': 0.8,         # 80% CPU usage threshold
    # 'pod_processes': 2,        # Maximum number of processes
    
    # Infrastructure Node metrics (usage rates use threshold)
    # 'node_cpu_usage_rate': 0.8,   # 80% CPU usage threshold
    # 'node_memory_usage_rate': 0.8, # 80% memory usage threshold
    # 'node_filesystem_usage_rate': 0.8  # 80% filesystem usage threshold
}
ML_METRICS = {
    'node_memory_usage_rate',
    'node_cpu_usage_rate'
}

# Metrics that should use 3-sigma method
SIGMA_METRICS = {
    # APM metrics (request/response counts use 3-sigma)
    'client_error_ratio',
    'error_ratio',
    'server_error_ratio',
    'request',      # Request count variations
    'response',     # Response count variations
    'timeout',      # Timeout count variations
    'pod_cpu_usage',
    'pod_processes', 
    
    # Infrastructure Pod metrics (I/O and network metrics use 3-sigma)
    'pod_fs_reads_bytes',
    'pod_fs_writes_bytes',
    'pod_network_receive_bytes',
    'pod_network_transmit_bytes',
    'pod_network_receive_packets',
    'pod_network_transmit_packets',
    'pod_memory_working_set_bytes',
    
    # Infrastructure Node metrics (I/O and network metrics use 3-sigma)
    'node_cpu_usage_rate',
    'node_memory_usage_rate',
    'node_filesystem_usage_rate',
    'node_disk_read_bytes_total',
    'node_disk_written_bytes_total',
    'node_disk_read_time_seconds_total',
    'node_disk_write_time_seconds_total',
    'node_filesystem_free_bytes',
    'node_filesystem_size_bytes',
    'node_memory_MemAvailable_bytes',
    'node_memory_MemTotal_bytes',
    'node_network_receive_bytes_total',
    'node_network_transmit_bytes_total',
    'node_network_receive_packets_total',
    'node_network_transmit_packets_total',
    'node_sockstat_TCP_inuse'
}

def _parse_and_convert_time(time_str: str) -> datetime:
    """
    Parse time string and convert to UTC time.
    Supports two formats:
    1. Beijing time format: '2025-05-05 12:11:09'
    2. UTC format: '2025-05-15T21:10:07Z'
    
    Args:
        time_str: Time string in either Beijing time or UTC format
        
    Returns:
        datetime object in UTC timezone
    """
    # Check if it's UTC format (ends with Z)
    if time_str.endswith('Z'):
        try:
            # Parse UTC time directly
            dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
            return pytz.UTC.localize(dt)
        except ValueError:
            raise ValueError(f"Invalid UTC time format: {time_str}")
    else:
        try:
            # Assume Beijing time format
            beijing_tz = pytz.timezone("Asia/Shanghai")
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            local_dt = beijing_tz.localize(dt)
            return local_dt.astimezone(pytz.UTC)
        except ValueError:
            raise ValueError(f"Invalid Beijing time format: {time_str}")

def _convert_to_utc(beijing_time_str: str) -> datetime:
    """Convert Beijing time string to UTC datetime object"""
    beijing_tz = pytz.timezone('Asia/Shanghai')
    utc_tz = pytz.UTC
    
    beijing_time = beijing_tz.localize(datetime.strptime(beijing_time_str, '%Y-%m-%d %H:%M:%S'))
    return beijing_time.astimezone(utc_tz)

def _convert_to_beijing(utc_time_str: str, delay: int = 0) -> datetime:
    """
    Convert UTC time string to Beijing datetime object
    
    Args:
        utc_time_str: Time string in UTC format (e.g. '2025-05-15T21:10:07Z')
        delay: Time shift in minutes (default: 0)
    """
    utc_tz = pytz.UTC
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    if utc_time_str.endswith('Z'):
        dt = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%SZ")
    else:
        dt = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%S")
    
    utc_dt = utc_tz.localize(dt)
    beijing_dt = utc_dt.astimezone(beijing_tz)
    
    if delay != 0:
        beijing_dt = beijing_dt + timedelta(minutes=delay)
        
    return beijing_dt

def _beijing_to_unix_seconds(beijing_time: datetime | str) -> int:
    """
    Convert Beijing time to Unix timestamp in seconds.

    Args:
        beijing_time: Beijing time as datetime or string (e.g. '2025-05-05 12:11:09')

    Returns:
        int: Unix timestamp in seconds
    """
    beijing_tz = pytz.timezone('Asia/Shanghai')
    if isinstance(beijing_time, datetime):
        if beijing_time.tzinfo is None:
            local_dt = beijing_tz.localize(beijing_time)
        else:
            local_dt = beijing_time.astimezone(beijing_tz)
    else:
        dt = datetime.strptime(beijing_time, '%Y-%m-%d %H:%M:%S')
        local_dt = beijing_tz.localize(dt)
    return int(local_dt.timestamp())

def _get_date_range(start_utc: datetime, end_utc: datetime) -> List[str]:
    """
    Get all dates within the UTC time range
    
    Args:
        start_utc: UTC start time
        end_utc: UTC end time
    
    Returns:
        List of dates in YYYY-MM-DD format
    """
    dates = []
    current_dt = start_utc
    while current_dt <= end_utc:
        dates.append(current_dt.strftime('%Y-%m-%d'))
        # Ensure no data is missed between days
        current_dt = (current_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return sorted(list(set(dates)))  # Remove duplicates and sort

def _get_instance_type(instance: str) -> str:
    """
    Determine instance type based on instance name
    
    Returns:
        'pod', 'service', or 'node'
    """
    # Check if it's a node
    if (instance.startswith('aiops-k8s-') and re.match(r'aiops-k8s-0[1-8]$', instance)) or \
       (instance.startswith('k8s-master') and re.match(r'k8s-master[1-3]$', instance)):
        return 'node'
    
    # Check if it's a pod (format: name-number, number is 0-2)
    if re.match(r'.*-[0-2]$', instance):
        return 'pod'
    
    # Check if it's a TiDB related pod
    if instance in ["tidb-tidb", "tidb-pd", "tidb-tikv"]:
        return 'pod'

    
    # Check if it's a service (ends with 'service')
    if (
        instance.endswith("service")
        or instance == "frontend"
        or instance.startswith("redis-")
    ):
        return "service"
    
    # If no match, raise exception
    raise Exception("Invalid instance name.")

def _get_metric_type(metric_name: str) -> str:
    """
    Determine metric type based on metric name

    Returns:
        'apm', 'infra_pod', or 'infra_node' 'infra_tidb' 'infra_tikv' 'infra_pd'
    """
    if metric_name in apm_metric_names:
        return "apm"
    elif metric_name in infra_pod_metric_names and metric_name.startswith("pod_"):
        return "infra_pod"
    elif metric_name in infra_node_metric_names:
        return "infra_node"
    # tidb, tikv, pd metrics
    elif metric_name in infra_tidb_metric_names:
        return "infra_tidb"
    elif metric_name in tidb_tikv_metric_names:
        return "tidb_tikv"
    elif metric_name in tidb_pd_metric_names:
        return "tidb_pd"

    # If no match, raise exception
    raise Exception("Invalid metric name.")


def _detect_anomalies_threshold(values: np.ndarray, metric_name: str) -> Tuple[np.ndarray, dict]:
    """
    Detect anomalies using threshold method
    
    Args:
        values: Array of metric values
        metric_name: Name of the metric
    
    Returns:
        Tuple containing anomaly indices and statistics
    """
    threshold = THRESHOLD_METRICS.get(metric_name)
    if threshold is None:
        raise ValueError(f"No threshold defined for metric: {metric_name}")
    
    anomaly_indices = np.where(values > threshold)[0]
    
    stats = {
        'threshold': threshold,
        'anomaly_indices': anomaly_indices,
        'anomaly_percentage': len(anomaly_indices) / len(values) if len(values) > 0 else 0
    }
    
    return anomaly_indices, stats

def _detect_anomalies_sigma(values: np.ndarray, baseline_values: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Detect anomalies using 3-sigma method
    
    Args:
        values: Array of metric values to check
        baseline_values: Array of baseline values for calculating statistics
    
    Returns:
        Tuple containing anomaly indices and statistics
    """
    if len(baseline_values) == 0:
        return np.array([]), {
            'mean': 0,
            'std': 0,
            'upper_bound': 0,
            'lower_bound': 0,
            'anomaly_indices': [],
            'anomaly_percentage': 0
        }

    mean = np.mean(baseline_values)
    std = np.std(baseline_values)
    upper_bound = mean + 5 * std
    lower_bound = mean - 5 * std
    
    anomaly_indices = np.where(
        (values > upper_bound) | (values < lower_bound)
    )[0]
    
    anomly_percentage = len(anomaly_indices) / len(values) if len(values) > 0 else 0
    if anomly_percentage < 0.1:
        anomly_percentage = 0
        anomaly_indices = []
    stats = {
        'mean': mean,
        'std': std,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'anomaly_indices': anomaly_indices,
        'anomaly_percentage': anomly_percentage
    }
    
    return anomaly_indices, stats

def _generate_anomaly_description(
    metric_name: str,
    stats: dict,
    values: np.ndarray,
    baseline_values: np.ndarray = None,
    instance: str = None
) -> str:
    """
    Generate anomaly description text
    
    Args:
        metric_name: Name of the metric
        stats: Anomaly detection statistics
        values: Values during anomaly period
        baseline_values: Values during baseline period (can be None for threshold detection)
        instance: Instance name (pod, service, or node)
    
    Returns:
        str: Anomaly description text
    """
    description = []
    instance_info = f"[{instance}] " if instance else ""
    
    # Get current statistics
    current_mean = np.mean(values) if len(values) > 0 else 0
    
    if stats['detection_method'] == 'threshold':
        threshold = stats['threshold']
        if len(stats['anomaly_indices']) > 0 and stats['anomaly_percentage'] > 0.05: # Need to be revise
            if metric_name.endswith('ratio'):
                description.append(f"{instance_info}Metric {metric_name} shows anomaly: current average is {current_mean:.2f}, exceeding threshold {threshold:.2f}")
            else:
                description.append(f"{instance_info}Metric {metric_name} shows anomaly: current average is {current_mean:.2f}, exceeding threshold {threshold}")
    elif stats['detection_method'] == '3-sigma':  # 3-sigma method
        if len(stats['anomaly_indices']) > 0:
            baseline_mean = stats['mean']
            deviation = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else float('inf')
            if deviation == float('inf') or deviation <= 25:
                return ""
                description.append(f"{instance_info}No anomalies detected for metric {metric_name}")
            if baseline_mean < 0.1 and current_mean < 0.1:
                return ""
            trend = "increased" if deviation > 0 else "decreased"
            
            # Generate different descriptions based on metric type
            if 'cpu' in metric_name: 
                description.append(f"{instance_info}CPU usage anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
            elif 'memory' in metric_name:
                description.append(f"{instance_info}Memory usage anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
            elif 'network' in metric_name:
                if 'receive' in metric_name:
                    description.append(f"{instance_info}Network receive traffic anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
                else:
                    description.append(f"{instance_info}Network transmit traffic anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
            elif metric_name == 'request':
                description.append(f"{instance_info}Request count anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
            elif metric_name == 'response':
                description.append(f"{instance_info}Response count anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
            elif metric_name == 'timeout':
                description.append(f"{instance_info}Timeout count anomaly detected: {abs(deviation):.2f}% {trend} compared to baseline period")
            else:
                description.append(f"{instance_info}Metric {metric_name} shows anomaly: {abs(deviation):.2f}% {trend} compared to baseline period")
            
            description.append(f"Current average: {current_mean:.2f}, Baseline average: {baseline_mean:.2f}")
            # description.append(f"Anomaly detection bounds: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
    elif stats['detection_method'] == 'FCVAE':  # FCVAE method
        description.append(f"{instance_info}Metric {metric_name} shows anomaly.\n")
    
    if len(stats['anomaly_indices']) > 0 and stats['anomaly_percentage'] > 0.05:
        description.append(f"Percentage of anomalous points: {stats['anomaly_percentage']*100:.2f}%")
    else:
        return ""
        description.append(f"{instance_info}No anomalies detected for metric {metric_name}")

    return "\n".join(description)