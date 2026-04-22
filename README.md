# 代码整合版（`代码整合版.py`）使用说明


## 快速开始

查看所有子命令：

```bash
python "代码整合版.py" --help
```

运行任一功能（示例）：

```bash
python "代码整合版.py" viz10
python "代码整合版.py" trend
```

端到端一键跑完（抽取合并→打标签→可视化→预测）：

```bash
export OPENAI_API_KEY="你的key"
# 可选：export OPENAI_BASE_URL="你的 OpenAI 兼容网关"
# 可选：export OPENAI_MODEL="deepseek-chat"

python "代码整合版.py" pipeline --month 12 --project-dir .
```

---

## 依赖安装（推荐）

你的环境里通常需要这些库（不同子命令依赖不同）：

```bash
python -m pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels
```

### 可选依赖

- **导出 Plotly PNG**（否则只会输出 HTML）：  
  - `kaleido`

```bash
python -m pip install kaleido
```

- **趋势预测 `trend`（LSTM）**：  
  - `tensorflow` / `keras`

```bash
python -m pip install tensorflow keras
```

> 说明：`trend` 会在运行时按需导入深度学习依赖；未安装时会提示你缺哪些包，不会影响其他子命令。

---

## 环境变量（LLM / pipeline）

`pipeline` 会调用 LLM 完成抽取、合并摘要/标题、以及打标签。你需要提供以下环境变量（或用命令行参数传入）：

- `OPENAI_API_KEY`：必填（也可用 `--api-key` 传入）
- `OPENAI_BASE_URL`：可选（也可用 `--base-url` 传入）
- `OPENAI_MODEL`：可选（也可用 `--model` 传入）

---

## 数据目录

`代码整合版.py` 会自动探测默认数据路径；你也可以随时用参数覆盖。

### 10 月（`viz10 / worldmap10 / lda10`）

默认按顺序尝试：

- `./10`
- `./data/10`
- `./dataset/10`
- `./5508 visual code/10`

### 11 月（`viz11 / heatmap11 / worldmap11 / lda11`）

默认按顺序尝试：

- `./11月数据_标签`
- `./data/11月数据_标签`
- `./dataset/11月数据_标签`
- `./5508 visual code/11`

### 12 月（`viz12 / worldmap12 / lda12`）

默认按顺序尝试：

- `./12月标签`
- `./data/12月标签`
- `./dataset/12月标签`
- `./5508 visual code/12`

### 趋势预测（`trend`）

默认按顺序尝试：

- `./data/new_analysis`
- `./dataset/new_analysis`
- `./new_analysis`
- `./data_2025_full/new_analysis`

### 端到端流水线（`pipeline`）

默认读取项目目录下的原始新闻数据：

- `./data_2025_full/news/news_2025_{month}.xlsx`（month 取 10/11/12）

---

## 子命令说明与输出

### `viz10`：10 月整套可视化（4 张图）

```bash
python "代码整合版.py" viz10 --base-path "/你的/10月数据目录"
```

需要的文件（放在 `--base-path` 目录下）：

- `1. sms_scam_types.xlsx`
- `2. news_scam_types.xlsx`
- `3. social_media_scam_types.xlsx`
- `1. news_scam_cases_2025_10.xlsx`

输出文件（同目录）：

- `viz_1_primary_type_by_source.png`
- `viz_2_news_location_vs_type.png`
- `viz_3_news_amount_by_type_clean.png`
- `viz_4_world_map_news_scams.html`（以及可选 `viz_4_world_map_news_scams.png`）

---

### `worldmap10`：10 月全球分布图（优化版）

```bash
python "代码整合版.py" worldmap10 --base-path "/你的/10月数据目录"
```

输出：

- `viz_4_world_map_news_scams.html`（可选 PNG）

---

### `heatmap10-clean`：无需数据文件的英文热图

```bash
python "代码整合版.py" heatmap10-clean
python "代码整合版.py" heatmap10-clean --output "./Clean_English_Scam_Heatmap_Final.png"
```

---

### `lda10 / lda11 / lda12`：话术 LDA 主题建模

示例：

```bash
python "代码整合版.py" lda10 --base-path "/你的/10月话术目录" --topics 5
python "代码整合版.py" lda11 --base-path "/你的/11月目录" --topics 5
python "代码整合版.py" lda12 --base-path "/你的/12月目录" --month 12 --topics 5
```

输出：在对应目录下生成 `xlsx` 报告（主题概览 + 明细分类）。

---

### `viz11 / heatmap11 / worldmap11`：11 月可视化

```bash
python "代码整合版.py" viz11 --base-path "/你的/11月数据_标签目录"
python "代码整合版.py" heatmap11 --base-path "/你的/11月数据_标签目录"
python "代码整合版.py" worldmap11 --base-path "/你的/11月数据_标签目录"
```

输出（示例）：

- `viz_1_scam_type_distribution.png`
- `Nov_Comparison_Heatmap_Fixed.png`
- `viz_map_split_nov.html`

---

### `viz12 / worldmap12`：12 月可视化

```bash
python "代码整合版.py" viz12 --base-path "/你的/12月标签目录"
python "代码整合版.py" worldmap12 --base-path "/你的/12月标签目录"
```

输出（示例）：

- `12月_诈骗类型热图.png`
- `12月_类型柱状图.png`
- `12月_金额分布.png`（若有金额列）
- `12月_全球诈骗分布地图.html`

---

### `trend`：诈骗趋势预测（ARIMA/LSTM）

```bash
python "代码整合版.py" trend --data-dir "/你的/new_analysis目录" --output-dir "./"
```

输入文件（放在 `--data-dir` 目录下，至少需要其中之一；建议齐全）：

- 新闻：`news_2025_10_extracted.xlsx`、`news_2025_11_extracted.xlsx`、`news_2025_12_extracted.xlsx`
- 社媒：`social_media_patterns_2025_10.xlsx`、`social_media_patterns_2025_11.xlsx`、`social_media_patterns_2025_12.xlsx`

输出（默认当前目录）：

- `1_total_comparison_smooth.png`
- `2_type_forecast.png`
- `3_tactic_forecast.png`
- `4_region_forecast.png`
- `5_platform_forecast.png`
- `prediction_results.txt`

并且在代码里按流程标注了：`step1` 到 `step8`（便于展示/答辩）。

---

### `pipeline`：端到端（抽取合并→打标签→可视化→预测）

```bash
python "代码整合版.py" pipeline --month 12 --project-dir .
```

参数：

- `--api-key/--base-url/--model`：LLM 配置（也可用环境变量）
- `--output-dir`：输出目录（默认 `project_dir/pipeline_out_{month}`）
- `--no-viz`：不运行 step6（可视化）
- `--no-forecast`：不运行 step7（预测）

流程（已在代码里显式标注 step1..step7）：

- **step1**：新闻案件信息抽取 → `news_2025_{month}_extracted.xlsx`
- **step2**：按 `case_key` 聚类合并案件 → `scam_cases_final_2025_{month}.xlsx`
- **step3**：Scam types 打标（primary/secondary + scam_process）→ `news_scam_types.xlsx`
- **step4**：Scam tactics 打标（tactic_categories）→ `news_tactic_categories_ai.xlsx`
- **step5**：script_pattern 抽取（3–6步流程）→ `news_with_scripts.xlsx`
- **step6**：pipeline 专用可视化（不依赖旧脚本命名）→ `pipeline_step6_*.png`
- **step7**：pipeline 专用预测（基于 publish_time 构建日序列，ARIMA；可选 LSTM）→ `pipeline_step7_*.png` + `pipeline_step7_results.txt`

输出目录（默认）：`./pipeline_out_{month}/`

---

## 常见问题

### 1) 报错提示缺少 Excel 文件

这是正常的“输入检查”。把提示里列出的文件放到默认目录，或显式传入 `--base-path/--data-dir`。

### 2) Plotly 导不出 PNG，只生成了 HTML

安装 `kaleido` 后重跑即可：

```bash
python -m pip install kaleido
```

### 3) `trend` 提示缺少 tensorflow/keras

安装后再运行：

```bash
python -m pip install tensorflow keras
```

### 4) `pipeline` 提示未提供 API key

设置环境变量后重跑：

```bash
export OPENAI_API_KEY="你的key"
python "代码整合版.py" pipeline --month 12 --project-dir .
```

