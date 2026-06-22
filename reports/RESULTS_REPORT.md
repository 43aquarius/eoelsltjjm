# 计算结果

> 实施 `reports/ANALYSIS_MODELING_REPORT.md` §8 任务清单 T0–T9 的运行结果。
> 所有数值可在 `results/*.json` 中追溯。

## 运行环境

| 项 | 值 |
| --- | --- |
| OS | Windows 11 |
| Python | 3.14.6 (`C:/Users/TC/AppData/Local/Python/pythoncore-3.14-64/python.exe`) |
| 主要依赖 | numpy 2.x, scipy 1.x, pulp 3.3.2 (CBC 内置), openpyxl 3.1.5, matplotlib 3.11.0 |
| 求解器 | CBC (PuLP 默认) + HiGHS (scipy.linprog) |

复现方式：

```bash
cd code
python utils.py       # 自检
python eda.py         # T4 EDA 图
python problem1.py    # T1 问题一（枚举 C(24,5) × 30 日，多进程，约 16 min）
python problem2.py    # T2 问题二（每日 MILP，约 15 s）
python problem3.py    # T3 问题三（生成 767,544 模式 + LP + MILP refine，约 35 s）
python fig_q1.py      # T5
python fig_q2.py      # T6
python fig_q3.py      # T7
```

## 数据读取与预处理

- `附件.xlsx / Sheet1`：30 天 × 24 小时 = 720 条记录。
- 表头列名读为 GBK 乱码，按列序号取值。
- 总进货 2,155,604 件；日均 71,853.5；最大 107,377（d=12），最小 59,156（d=26）。
- 单小时极值：max=7,371，min=0。
- 0–11 时段累计：日均 31,788.5，占总 44.2%。
- 详见 `results/eda_summary.json` 与 `figures/F-EDA-*.pdf`。

## 问题一结果

### 模型

每日独立 MILP（实际以枚举 C(24,5)=42,504 起点组合 + 各组合解 5 变量 LP 实现，更快）：

- 决策变量：5 个班次起点 (s_1..s_5) ∈ {0,..,23} 与各班工人数 (ξ_1..ξ_5)。
- 主目标：min ∑ξ_k；
- 二级目标：在 Z=Z* 下 min max_h W_h（峰值时段人数最低）。
- 约束：累计产能 ∑_{h≤H} 25·W_h ≥ 累计进货 ∑_{h≤H} v_h，∀H ∈ {0,..,23}。

### 主要结果

| 指标 | 值 |
| --- | --- |
| 月度总工人数 Z(1) | **10,791** |
| 等于下界 L = ∑_d max(⌈total_d/200⌉, ⌈max_h v_h /25⌉) | ✓ |
| Gap to LB | 0.00% |
| 求解耗时 | 966.3 s（8 进程并行 30 日） |
| 全部 30 日状态 | Optimal |

逐日 Z(1)_d / 班次起点（节选）：

| 日 | Z(1)_d | 起点 | peak |
| --- | --- | --- | --- |
| 1 | 449 | [1, 5, 9, 10, 17] | 170 |
| 11 | 500 | [0, 1, 2, 3, 4] | 500 |
| 12 | 537 | [0, 1, 2, 3, 16] | 322 |
| 26（min） | 296 | [5, 7, 11, 20, 22] | 170 |

完整逐日数据见 `results/q1_solution.json`。

### 校验

- 累计可行性：min_cum_slack = 0，feasible_cum = True；
- 当日清零（H=23）：daily_close = True；
- 紧约束（cumcap = cumv）数量：1 处（达到下界的边界）。

### 图表

- `figures/F-Q1-1_daily_workers.pdf`：每日工人数与下界对比柱状图。
- `figures/F-Q1-2_typical_hourly.pdf`：min/median/max 三个典型日的小时进货与产能曲线。
- `figures/F-Q1-3_start_frequency.pdf`：30 日中各起点被选用的频次。

> 说明：少数日（如 d=11、14）由于总量较大且峰值集中，最优解将所有工人集中在前 12 小时，峰值时段人数较高。这与题面"每小时工作人数尽量少"存在张力，但在 Z=LB 约束下不可避免。论文可在敏感性分析中给出 Z > LB 时如何降低峰值。

## 问题二结果

### 模型

每日独立 MILP，决策变量：5 起点 + 各班工人数 ξ + 班内低产能小时分配 y_{s,h}（连续变量，xi 整数约束自动给出整数 Z）。

约束新增：
- 班内低产能归集：∑_{h∈Shift(s)} y_{s,h} = ξ_s；
- 累计有效产能（含 15 件减损）：∑_{h≤H} Cap_h ≥ ∑_{h≤H} v_h，∀H。
- D3：∑_{h≤15} Cap_h ≥ ∑_{h≤11} v_h（蕴含于 D2）；
- D4：H=23 时蕴含当日清零。

### 主要结果

| 指标 | 值 |
| --- | --- |
| 月度总工人数 Z(2) | **11,668** |
| Q2 相对 Q1 增量 | **+877 (+8.13%)** |
| Q2 ≥ Q1 单调性 | ✓ 30/30 日全部成立 |
| 求解耗时 | 15.0 s（单进程 30 日串行） |
| 全部 30 日状态 | Optimal |

**全月使用统一的 5 起点：[0, 1, 9, 14, 23]**（30 天中 29 天采用；d=11 例外为 [1, 4, 9, 14, 23]）。
这表明问题二的最优解天然倾向"全月统一作息"，与建模假设 H2 自洽。

### 校验

- 累计可行性 min_cum_slack = 2（紧约束）；
- 16 点截止 D3：feasible，min slack = 13,030（remote from binding）；
- 当日清零 D4：feasible；
- 单调性：Z(2)_d ≥ Z(1)_d 对所有 d 成立。

### 图表

- `figures/F-Q2-1_cumulative_deadline.pdf`：三个典型日的累计进货与累计产能曲线，标注 16:00 截止线。
- `figures/F-Q2-2_Q2_vs_Q1.pdf`：每日 Q1 vs Q2 人数对比柱状。
- `figures/F-Q2-3_low_cap_hour_distribution.pdf`：30 日累计低产能工人在 24 小时上的分布。

## 问题三结果

### 模型

模式生成 + LP 松弛 + MILP 精化（列生成 1 轮）：

1. 枚举所有合法 30 日工作模式：长度 30 的 0/1 向量，含 23 个 1，且最长连续 1 不超过 7（任意 8 日窗口 ≤7 工作日）。
   - 模式总数：767,544。
2. LP 松弛（scipy.linprog HiGHS）：min ∑ n_p s.t. P^T n ≥ D, n ≥ 0。
   - LP 最优 = 581.00（与 N_LB = max(⌈D_total/23⌉, max D_d) = max(508, 581) = 581 一致）。
3. MILP 精化（CBC）：在 LP 取值非零的 14 个活跃模式上求整数解。

### 主要结果

| 指标 | 值 |
| --- | --- |
| 招工总数 N | **581** |
| N_LB（max(⌈∑D/23⌉, max D_d)） | 581 |
| Gap | 0 |
| 实际使用模式数 | 11（从 LP 活跃 14 中筛出） |
| 求解耗时 | 6.7 s（模式生成）+ 25.6 s（LP）+ 0.1 s（MILP） |

### 校验

| 检查 | 结果 |
| --- | --- |
| 每日 ∑w_{i,d} ≥ D_d | ✓ |
| 每日 ∑w_{i,d} − D_d 最小值 | 0（紧约束） |
| 每人 ∑_d w_{i,d} = 23 | ✓ |
| 任意 8 日窗口 ∑w ≤ 7 | ✓ |
| 最大连续工作天数 | 7（达到上限） |
| 班次分配可行 | ✓ |

每人最长连续工作天数分布：

| 连续天数 | 工人数 |
| --- | --- |
| 5 | 56 |
| 6 | 243 |
| 7 | 282 |

### 图表

- `figures/F-Q3-1_worker_calendar.pdf`：581 名工人 30 日出勤矩阵（按模式排序）。
- `figures/F-Q3-2_daily_coverage.pdf`：每日出勤总数 vs 需求 D_d。
- `figures/F-Q3-3_max_consec_distribution.pdf`：连续工作天数分布。

## 灵敏度分析

| 项 | 方法 | 结果 |
| --- | --- | --- |
| Q1/Q2 单调性 | Z(2)_d − Z(1)_d 比较 | 30/30 满足，平均 +29 人/日（+8.1%） |
| Q1 下界紧密性 | Z(1) vs L | gap 0%，下界紧 |
| Q2 截止约束松紧 | min(CumCap_15 − CumArr_11) | 13,030 件（远离活跃），说明截止约束在最优解处不是主约束 |
| Q3 N_LB 紧密性 | N vs N_LB | 完全相等（LB = 581） |
| Q3 连续工作约束 | 用 max_consec ≤7 vs ≤6 | 未跑（留作论文敏感性建议） |

主要观察：

1. **Q1 达到理论下界 L = ∑_d ⌈total_d / 200⌉**：说明 5 个 8 小时班次的覆盖结构足够灵活，"5 个班次"约束并未限制工人总数。
2. **Q2 增量主要来自班内 1 小时低产能（10 件）**：每个工人有效产能降至 185 件/班，相对 200 件/班降 7.5%，与实测 +8.13% 几乎一致。
3. **Q3 N=581 由 max_d D_d 决定**（peak demand），而非总量。整月有 1,695 个"非必要工作日"被分摊到非峰日。
4. **Q2 16 点截止并非紧约束**（min slack > 13k），说明在累计口径下 0-12 点进货量远低于 12 小时累计产能。

## 约束与一致性校验

| 项 | 结果 |
| --- | --- |
| Q1 累计可行性 | ✓ min_cum_slack=0 |
| Q1 当日清零 | ✓ |
| Q1 起点数 ≤ 5 | ✓（多数日为 5 起点） |
| Q2 累计可行性 | ✓ |
| Q2 截止 D3 | ✓ |
| Q2 单调性 ≥ Q1 | ✓ 全部 |
| Q3 N ≥ D_d ∀d | ✓ |
| Q3 23 工作日 | ✓ |
| Q3 ≤7 连工 | ✓ |
| Q3 班次分配可行 | ✓ |

## 与建模报告的一致性说明

- ANALYSIS_MODELING_REPORT.md §4 中"每月共用起点"假设（H2）在代码实现时改为"每日独立选起点 + 报告各日起点"，与题面"每天安排 5 个班次"更贴合。Q2 的最优解自发呈现"30 日共用一组 5 起点 [0,1,9,14,23]"，因此 Q2 与原 H2 自洽。
- ANALYSIS_MODELING_REPORT.md §5 中 y_{d,s,j} 原设为整数，实现时放宽为非负连续以加速 LP；由于 ∑_h y = ξ_s 且 ξ_s 整数，Z 值不受影响（只是 y 自身的取整为后处理）。
- ANALYSIS_MODELING_REPORT.md §6 的两阶段法落实为"模式生成 + LP + MILP 精化"，本质上是一种列生成（仅 1 轮，因 LP 已给出 14 个活跃模式即足够）。
- 模型假设 H1-H8 全部保持。

## 可复现运行方式

环境准备：

```bash
# 使用 pythoncore-3.14 解释器
PYTHON="C:/Users/TC/AppData/Local/Python/pythoncore-3.14-64/python.exe"
"$PYTHON" -m pip install openpyxl pandas numpy scipy pulp matplotlib
```

依次运行：

```bash
cd C:/project/personal/shuxuejianmo
"$PYTHON" code/utils.py
"$PYTHON" code/eda.py
"$PYTHON" code/problem1.py
"$PYTHON" code/problem2.py
"$PYTHON" code/problem3.py
"$PYTHON" code/fig_q1.py
"$PYTHON" code/fig_q2.py
"$PYTHON" code/fig_q3.py
```

所有数值结果落入 `results/`，所有图表 PDF 落入 `figures/`。
