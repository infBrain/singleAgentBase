# Evaluation

Score MCP and RCA outputs independently using common RCA ranking metrics.

## Usage

```bash
python src/eval/evaluate_results.py \
  --mcp eval_result/eval_mcp/mcp_mcp_processing_result_0127.json \
  --rca eval_result/eval_mcp/rca_rca_results_20260125_063144.json
```

Optional arguments:
- `--output`: custom output path

## Output

The output JSON contains:
- `mcp`: MCP scoring summary
- `rca`: RCA scoring summary

Each summary includes:
- `stats`: counts (total, with_ground_truth, with_locations, hit@K)
- `rates`: hit@K, coverage, precision/recall/F1, MRR, MAP, NDCG
- `misses`: cases where ground truth was not hit by any predicted location

## Metrics

### Stats
- `total`: Total number of entries in the dataset.
- `with_ground_truth`: Number of entries that have a non-empty `ground_truth` list.
- `with_locations`: Number of entries that have at least one predicted location.
- `hit_at_any`: Number of entries where at least one predicted location matches any ground truth location.
- `hit_at_1`: Number of entries where the first predicted location matches any ground truth location.
- `hit_at_3`: Number of entries where any of the top 3 predicted locations match any ground truth location.
- `hit_at_5`: Number of entries where any of the top 5 predicted locations match any ground truth location.

### Rates
- `hit_at_any_rate`: Proportion of entries with ground truth that have at least one hit (hit_at_any / with_ground_truth).
- `coverage_rate`: Proportion of entries that have predicted locations (with_locations / total).
- `hit_at_1_rate`: Proportion of entries with ground truth that have a hit in the first prediction (hit_at_1 / with_ground_truth).
- `hit_at_3_rate`: Proportion of entries with ground truth that have a hit in the top 3 predictions (hit_at_3 / with_ground_truth).
- `hit_at_5_rate`: Proportion of entries with ground truth that have a hit in the top 5 predictions (hit_at_5 / with_ground_truth).
- `precision`: Average precision across entries (true positives / predicted locations, averaged over all entries).
- `recall`: Average recall across entries (true positives / ground truth locations, averaged over all entries).
- `f1`: Average F1 score across entries (harmonic mean of precision and recall, averaged over all entries).
- `mrr`: Mean Reciprocal Rank: Average of 1/rank where rank is the position of the first correct location (averaged over entries with ground truth).
- `map`: Mean Average Precision: Average of average precision per entry (considering ranked predictions).
- `ndcg`: Normalized Discounted Cumulative Gain: Average NDCG score per entry (considering ranked predictions and ideal ranking).
