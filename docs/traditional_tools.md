# traditional_tools 工具说明

以下为 `src/tools/traditional_tools.py` 中工具方法说明，结构与 `docs/tools_list.txt` 类似。

---

name='get_logs'

描述：
获取离线 Parquet 中的时间序列日志（按时间排序）。

功能概述：
- 从离线数据中提取指定时间范围内的日志记录。
- 返回包含总数、截断数量和记录列表的 JSON。

使用场景：
- 本地/离线 RCA 分析
- 无法调用线上 MCP/SLS 时的回退路径

使用限制：
- 仅支持离线 Parquet 数据
- 时间范围过大可能导致返回数据量大

Args:
- start_time: 开始时间（ISO 8601，如 `2025-06-05T23:24:13Z`）
- end_time: 结束时间（ISO 8601）
- limit: 最大返回记录数（默认 200）

Returns:
- JSON 字符串，包含 `count`/`limited_count`/`records`

示例：
```
get_logs("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z", limit=100)
```

---

name='get_traces'

描述：
获取离线 Trace 汇总（按 service/pod 汇总的平均耗时与数量）。

功能概述：
- 从离线数据中提取 Trace 统计信息。
- 返回包含总数、截断数量和记录列表的 JSON。

Args:
- start_time: 开始时间（ISO 8601）
- end_time: 结束时间（ISO 8601）
- limit: 最大返回记录数（默认 200）

Returns:
- JSON 字符串，包含 `count`/`limited_count`/`records`

示例：
```
get_traces("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z")
```

---

name='get_metrics'

描述：
获取离线指标原始值。

功能概述：
- 返回指定指标在时间范围内的原始数值。
- 支持按指标名和实例过滤。

Args:
- start_time: 开始时间（ISO 8601）
- end_time: 结束时间（ISO 8601）
- metric_name: 指标名（默认 `all`）
- instance: 实例名（service/pod/node，默认 `all`）

Returns:
- JSON 字符串，包含 `count`/`limited_count`/`values`

示例：
```
get_metrics("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z", metric_name="cpu", instance="frontend")
```

---

name='detect_metrics'

描述：
进行离线时序指标异常检测。

功能概述：
- 基于离线指标进行异常检测分析。
- 支持按指标名和实例过滤。

Args:
- start_time: 开始时间（ISO 8601）
- end_time: 结束时间（ISO 8601）
- metric_name: 指标名（默认 `all`）
- instance: 实例名（service/pod/node，默认 `all`）

Returns:
- 检测结果（字符串）

示例：
```
detect_metrics("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z", metric_name="memory")
```

---

name='detect_traces'

描述：
进行离线 Trace 异常检测。

功能概述：
- 对离线 trace 数据进行异常分析。

Args:
- start_time: 开始时间（ISO 8601）
- end_time: 结束时间（ISO 8601）

Returns:
- 检测结果（字符串）

示例：
```
detect_traces("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z")
```

---

name='detect_logs'

描述：
进行离线日志异常检测。

功能概述：
- 对离线日志进行异常分析。

Args:
- start_time: 开始时间（ISO 8601）
- end_time: 结束时间（ISO 8601）

Returns:
- 检测结果（字符串）

示例：
```
detect_logs("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z")
```

---

name='get_system_info'

描述：
获取系统拓扑与配置信息。

功能概述：
- 返回系统调用关系、部署信息、配置细节等静态结构信息。

Args:
- 无

Returns:
- 拓扑与配置信息文本

示例：
```
get_system_info()
```

---

name='analyze_fault_type'

描述：
基于离线指标与检测结果推断故障类型。

功能概述：
- 使用内置分类器识别故障类别（如 Pod Memory、Network Delay）。

Args:
- start_time: 开始时间（ISO 8601）
- end_time: 结束时间（ISO 8601）

Returns:
- 故障类型分析结果（字符串）

示例：
```
analyze_fault_type("2025-06-05T23:24:13Z", "2025-06-05T23:40:13Z")
```
