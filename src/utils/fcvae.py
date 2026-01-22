import sys

sys.path.append("./")
sys.path.append("../")
from typing import Annotated, List, Tuple
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import re
from utils.common_utils import (
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
from utils.common_utils import (
    apm_metric_names,
    infra_node_metric_names,
    infra_pod_metric_names,
)
from utils.llm_chat import chat
from utils.multimodal_data import get_metric_values_offline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.fcvae_model import FCVAE
import matplotlib.pyplot as plt
WINDOW_SIZE = 128

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
    if "all" in instance.lower() and "all" in metric_name.lower():
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
        ]
        all_pod = []
        for service in all_service:
            for i in ["-0", "-1", "-2"]:
                all_pod.append(service + i)
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
                except:
                    print(f"Error in TSAD: {metric} {pod}")
                    continue
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "apm":
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
        results = []
        for service in all_service:
            results.append(
                time_series_anomaly_detection(
                    start_time, end_time, metric_name, service
                )
            )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_pod":
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
        results = []
        for pod in all_pod:
            results.append(
                time_series_anomaly_detection(start_time, end_time, metric_name, pod)
            )
        return "\n".join([item for item in results if item != ""])
    elif "all" in instance.lower() and _get_metric_type(metric_name) == "infra_node":
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
        results = []
        for node in all_nodes:
            results.append(
                time_series_anomaly_detection(start_time, end_time, metric_name, node)
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

    # For threshold-based detection, we don't need baseline data
    if metric_name in THRESHOLD_METRICS:
        baseline_values = np.array([])
        anomaly_indices, stats = _detect_anomalies_threshold(
            anomaly_values, metric_name
        )
    else:
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


def _detect_anomalies_fcvae(values: np.ndarray, baseline_values: np.ndarray) -> Tuple[np.ndarray, dict]:
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
            "mean": float(np.mean(baseline_values)) if baseline_values.size > 0 else 0.0,
            "std": float(np.std(baseline_values)) if baseline_values.size > 0 else 0.0,
            "upper_bound": None,
            "lower_bound": None,
        }

    # 窗口大小 W：history minutes = W - 1
    window_size = WINDOW_SIZE

    combined = np.concatenate([baseline_values, values], axis=0).astype(np.float32)
    total_len = combined.size
    if total_len < window_size:
        # 样本不足，直接返回无异常
        return np.array([], dtype=int), {
            "anomaly_indices": np.array([], dtype=int),
            "anomaly_percentage": 0.0,
            "prob_threshold": None,
            "recon_log_prob": np.array([], dtype=float),
            # 兼容描述逻辑
            "mean": float(np.mean(baseline_values)) if baseline_values.size > 0 else 0.0,
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
        log_prob = -0.5 * (torch.log(2 * torch.pi * var_last) + (x_last - mu_last) ** 2 / var_last)
        log_prob = log_prob.detach().cpu().numpy()

        # 可视化：每个窗口最后一个点的原始值与重建均值（标准化尺度）
        orig_last = x_last.detach().cpu().numpy()
        recon_last = mu_last.detach().cpu().numpy()
        try:
            plt.figure(figsize=(10, 4))
            plt.plot(orig_last, label="original(last)")
            plt.plot(recon_last, label="recon_mu(last)")
            plt.xlabel("window index")
            plt.ylabel("value (z-scored per window)")
            plt.title("Reconstruction vs Original (last point per window)")
            plt.legend()
            out_path = os.path.join(os.path.dirname(__file__), "reconstruction_last.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved plot to {out_path}")

            # 将最后一点从 z-score 还原回原始尺度：x = x_norm * std + mu
            std_np = np.squeeze(std, axis=1)
            mu_np = np.squeeze(mu, axis=1)
            orig_last_denorm = orig_last * std_np + mu_np
            recon_last_denorm = recon_last * std_np + mu_np

            plt.figure(figsize=(10, 4))
            plt.plot(orig_last_denorm, label="original(last, denorm)")
            plt.plot(recon_last_denorm, label="recon_mu(last, denorm)")
            plt.xlabel("window index")
            plt.ylabel("value (original scale)")
            plt.title("Reconstruction vs Original (denormalized, last point per window)")
            plt.legend()
            out_path_denorm = os.path.join(os.path.dirname(__file__), "reconstruction_last_denorm.png")
            plt.tight_layout()
            plt.savefig(out_path_denorm, dpi=150)
            plt.close()
            print(f"Saved plot to {out_path_denorm}")
        except Exception as _:
            pass

    prob_threshold = 0.8
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

# c = np.linspace(0, 1, 256)
# print("ccc",len(c),c)
# a, b = _detect_anomalies_fcvae( c[127:],c[:127])
# print(a, b)
