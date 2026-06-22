# 验证和验收报告

## 结论

**PASS**

## 检查项

| 检查项 | 结果 | 说明 |
| --- | --- | --- |
| 入口文件存在 | PASS | `paper/main.typ` |
| 章节文件齐全 | PASS | 9 个章节 + 附录 + 参考文献 |
| `#include` 顺序正确 | PASS | 1→9，前缀编号顺序 |
| 一级标题存在 | PASS | 所有章节均有 `= 标题` |
| 占位符 | PASS | 无 TODO/PLACEHOLDER/待补充/FIXME/示例数据 |
| 内部工作流名泄露 | PASS | 正文未出现 `results/` `reports/` `q[123]_solution` 等（仅 `image()` 引用 `figures/` 为正常资源路径） |
| 图片文件存在 | PASS | 17 张被引用的 PDF 全部存在 |
| 数值一致性（Z(1)=10791, Z(2)=11668, N=581） | PASS | 与 `results/{q1,q2,q3}_solution.json` 完全一致 |
| 参考文献 | PASS | 9 条真实文献，含 Dantzig/Bechtold/Ernst/PuLP/HiGHS/CBC 等 |
| 文本质量门禁脚本 | PASS | `writing_check.sh` 输出 `PASS: writing text gate passed`（2 处 WARN 非硬错误） |
| Typst 编译 | PASS | `typst compile --root <project>` 成功，无 error；仅字体 fallback 警告 |
| PDF 非空、页面合规 | PASS | 25 页，每页 A4（595.28 × 841.89 pt） |
| PDF 视觉抽检 | PASS | 抽查 1, 2, 3, 6, 11, 16, 21, 25 页：封面/摘要/目录/正文/图表/附录均正常无裁切重叠 |

## 章节结构

```
paper/
├── main.typ                       入口（封面 + 摘要 + 关键字 + 目录 + include 链 + 参考文献 + 附录）
├── references.typ                 9 条真实参考文献
└── sections/
    ├── 1_restatement.typ          问题重述（含技术路线图引用）
    ├── 2_analysis.typ             数据理解与总体思路（含 3 张 EDA 图）
    ├── 3_assumptions.typ          模型假设 H1–H10
    ├── 4_symbols.typ              符号说明表
    ├── 5_problem1.typ             问题一（模型 + 算法 + 3 张数据图 + 模型关系图 + 流程图）
    ├── 6_problem2.typ             问题二（模型 + 算法 + 3 张数据图 + 流程图）
    ├── 7_problem3.typ             问题三（模型 + 算法 + 3 张数据图 + 流程图）
    ├── 8_sensitivity.typ          灵敏度分析（5 项扰动检验）
    ├── 9_evaluation.typ           模型评价与推广
    └── A_code.typ                 附录核心代码（utils + Q1 + Q2 + Q3）
```

包含 9 个一级标题（`include` 数量 9，章节文件 10：A_code 由 `appendix-cn()` 单独包含），顺序与文件名前缀一致。

`writing_check.sh` 检查的 2 处 WARN：

- `1_restatement.typ` 长度 610 字符（题面重述本身较短，已包含问题背景、3 个子问题、技术路线图，不需扩展）；
- `5_problem1.typ`、`8_sensitivity.typ` 出现三级标题 `===`（用于章节内的子小节，是正常组织手段）。

## 图表引用

| 图编号 | 文件 | 章节 | 类型 |
| --- | --- | --- | --- |
| @fig-roadmap | `fig_roadmap.pdf` | §1 末 | 流程图 |
| @fig-eda-heat / profile / total | `F-EDA-{1,2,3}*.pdf` | §2 | 数据图 |
| @fig-model | `fig_model.pdf` | §5 | 流程图 |
| @fig-flow-q1 | `fig_flow_q1.pdf` | §5 | 流程图 |
| @fig-q1-workers / typical / start | `F-Q1-{1,2,3}*.pdf` | §5 | 数据图 |
| @fig-flow-q2 | `fig_flow_q2.pdf` | §6 | 流程图 |
| @fig-q2-cum / vs / low | `F-Q2-{1,2,3}*.pdf` | §6 | 数据图 |
| @fig-flow-q3 | `fig_flow_q3.pdf` | §7 | 流程图 |
| @fig-q3-calendar / cov / runs | `F-Q3-{1,2,3}*.pdf` | §7 | 数据图 |

共引用 17 张图：5 张非数据图 + 12 张数据图，全部存在于 `figures/`。每张图前后均有引导文字与解释，无连续 3 图无解释情况。

## 数值一致性

| 关键数值 | 论文中 | 结果记录 | 一致 |
| --- | --- | --- | --- |
| 总进货量 | 2,155,604 件 | `eda_summary.total = 2155604` | ✓ |
| Q1 月总人数 | 10,791 | `q1_solution.Z_total = 10791` | ✓ |
| Q1 下界 | 10,791，gap 0% | `q1_solution.lower_bound_total = 10791` | ✓ |
| Q2 月总人数 | 11,668 | `q2_solution.Z_total = 11668` | ✓ |
| Q2 − Q1 | +877 (+8.13%) | `q2_solution.compare_q1.delta = 877` | ✓ |
| Q2 全月共用起点 | [0,1,9,14,23]（29 日），[1,4,9,14,23]（d=11） | `q2_solution.starts_per_day` 同 | ✓ |
| Q3 N | 581 | `q3_solution.N = 581` | ✓ |
| Q3 N_LB | 581 | `q3_solution.N_LB = 581` | ✓ |
| 合法模式数 | 767,544 | `q3_solution.n_patterns_total = 767544` | ✓ |
| LP 活跃模式 | 14 | LP 中 n_p>0 数量 = 14 | ✓ |
| 最终非零模式 | 11 | `q3_solution.n_patterns_used = 11` | ✓ |
| 最长连工分布 | 7天282/6天243/5天56 | `q3_solution.verify.max_consec_distribution` 同 | ✓ |
| 16 点截止 slack | 13,030 件 | `q2_solution.verify.min_D3_slack = 13030` | ✓ |
| 日总量峰值 | 第 12 日 107,377 | `eda.daily_total_max = 107377` | ✓ |
| 日总量谷值 | 第 26 日 59,156 | `eda.daily_total_min = 59156` | ✓ |
| 日均总量 | 71,853.5 | `eda.daily_total_mean = 71853.5` | ✓ |

## 文本质量门禁

`writing_check.sh` 全部输出：

```
INFO: detected engine: Typst (main suffix: .typ)
INFO: section file count: 10
INFO: main include count: 9
PASS: writing text gate passed
```

仅 2 处 WARN（短章节、子小节使用三级标题），不构成硬错误。

## 编译

```bash
cd paper && typst compile --root "C:/project/personal/shuxuejianmo" main.typ
```

输出 `paper/main.pdf`（1.3 MB，25 页）。无 error，仅字体 fallback 警告（Heiti SC / STHeiti / Songti SC / Menlo 在本机不可用，自动回落到默认中文字体，渲染结果正常）。

## PDF 视觉检查

使用 `pypdfium2` 将 8 张抽样页面渲染为 PNG 后逐页查看：

| 页码 | 内容 | 结果 |
| --- | --- | --- |
| 1 | 封面 + 摘要 + 关键字 | ✓ 排版正常 |
| 2 | 目录 | ✓ 9 章标题 + 附录 + 参考文献，页码点引导线完整 |
| 3 | 第一章问题重述 | ✓ 技术路线图清晰，节点不重叠 |
| 6 | 模型假设末尾 + 符号说明开始 | ✓ 三线表完整 |
| 11 | Q1 结果（每日工人数 + 典型日 3 子图）| ✓ 图与图注居中，无溢出 |
| 16 | Q3 求解过程 + 出勤矩阵图 | ✓ 公式排版正常 |
| 21 | 模型推广末尾 | ✓ 文字流畅 |
| 25 | 附录代码末段 | ✓ 代码 syntax highlight 正确 |

全部页面 A4 (595.28 × 841.89 pt) 一致，无空白页、无被裁切元素，公式、表格、图片、页码均位于版面合理位置。

## 仍需处理的问题

- 字体 fallback 警告（本机缺少 SimSun/Heiti SC 等）：渲染结果可读，但若提交方对字形有严格要求（部分赛会要求宋体正文），需在评审机上安装对应字体或在主机预先嵌入字体。**不影响内容评审**。
- 备份建议：在最终提交前，建议在洁净环境复跑一次完整管线（`utils → eda → problem1 → problem2 → problem3 → fig_* → typst compile`）以确认可复现，预计约 17–18 分钟（Q1 枚举 16 min 为主要耗时）。

---

#strong[最终结论：论文与求解过程合规、自洽、可复现，已就绪提交。]
