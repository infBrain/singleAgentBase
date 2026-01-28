# Processing Cases 提取工具

从日志中提取 `Processing Cases` 与 `[HUMAN]` 之间的 JSON 片段，输出为一个 JSON 数组文件。

## 用法

```bash
python src/extract_processing_cases.py \
  -i result/mcp_result_0127.log \
  -o result/processing_cases.json \
  --errors result/processing_cases_errors.jsonl
```

## 输出

- `result/processing_cases.json`: JSON 数组，每个元素为一条 case 记录
- `result/processing_cases_errors.jsonl`: 可选，解析失败记录（每行一个 JSON）
