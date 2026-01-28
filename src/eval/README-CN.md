# 评估

使用常见的RCA排名指标独立评分MCP和RCA输出。

## 使用方法

```bash
python src/eval/evaluate_results.py \
  --mcp eval_result/eval_mcp/mcp_mcp_processing_result_0127.json \
  --rca eval_result/eval_mcp/rca_rca_results_20260125_063144.json
```

可选参数：
- `--output`: 自定义输出路径

## 输出

输出JSON包含：
- `mcp`: MCP评分摘要
- `rca`: RCA评分摘要

每个摘要包括：
- `stats`: 计数（总数、具有ground_truth的数量、具有locations的数量、hit@K）
- `rates`: hit@K、覆盖率、精确率/召回率/F1、MRR、MAP、NDCG
- `misses`: 预测位置未命中任何ground truth的情况

## 指标

### 统计
- `total`: 数据集中的条目总数。
- `with_ground_truth`: 具有非空`ground_truth`列表的条目数量。
- `with_locations`: 至少有一个预测位置的条目数量。
- `hit_at_any`: 至少有一个预测位置匹配任何ground truth位置的条目数量。
- `hit_at_1`: 第一个预测位置匹配任何ground truth位置的条目数量。
- `hit_at_3`: 前3个预测位置中任意一个匹配任何ground truth位置的条目数量。
- `hit_at_5`: 前5个预测位置中任意一个匹配任何ground truth位置的条目数量。

### 比率
- `hit_at_any_rate`: 具有ground truth的条目中至少有一个命中的比例（hit_at_any / with_ground_truth）。
- `coverage_rate`: 具有预测位置的条目比例（with_locations / total）。
- `hit_at_1_rate`: 具有ground truth的条目中第一个预测命中的比例（hit_at_1 / with_ground_truth）。
- `hit_at_3_rate`: 具有ground truth的条目中前3个预测中命中的比例（hit_at_3 / with_ground_truth）。
- `hit_at_5_rate`: 具有ground truth的条目中前5个预测中命中的比例（hit_at_5 / with_ground_truth）。
- `precision`: 条目间的平均精确率（真正例 / 预测位置，在所有条目上平均）。
- `recall`: 条目间的平均召回率（真正例 / ground truth位置，在所有条目上平均）。
- `f1`: 条目间的平均F1分数（精确率和召回率的调和平均，在所有条目上平均）。
- `mrr`: 平均倒数排名：第一个正确位置排名的1/rank的平均值（在具有ground truth的条目上平均）。
- `map`: 平均平均精确率：每个条目的平均精确率的平均值（考虑排名预测）。
- `ndcg`: 归一化折扣累积增益：每个条目的平均NDCG分数（考虑排名预测和理想排名）。