# DrawIO 图示生成报告

## 图示清单

| 文件 | 类型 | 来源依据 | 用途 | 状态 |
| --- | --- | --- | --- | --- |
| `fig_roadmap.{drawio,pdf}` | 技术路线图 | ANALYSIS_MODELING_REPORT §2.6 总体路线 + RESULTS_REPORT 三问结果 | 论文绪论 / 问题分析章节，呈现整体解题逻辑 | ✓ 双文件生成 |
| `fig_flow_q1.{drawio,pdf}` | 问题一求解流程图 | ANALYSIS_MODELING_REPORT §4 + problem1.py 实际实现（枚举 + LP + 双级目标） | 论文问题一章节 | ✓ |
| `fig_flow_q2.{drawio,pdf}` | 问题二求解流程图 | ANALYSIS_MODELING_REPORT §5 + problem2.py（MILP + 连续 y） | 论文问题二章节 | ✓ |
| `fig_flow_q3.{drawio,pdf}` | 问题三求解流程图 | ANALYSIS_MODELING_REPORT §6 + problem3.py（模式 + LP + MILP refine） | 论文问题三章节 | ✓ |
| `fig_model.{drawio,pdf}` | 模型变量关系图 | ANALYSIS_MODELING_REPORT §3 符号说明 + §4/§5 模型公式 | 论文符号说明 / 模型建立章节 | ✓ |

## 未生成图示及原因

- **数据处理 Pipeline 图**：本题数据简单（30×24 矩阵直接载入），EDA 流程已由 `fig_roadmap` 中"EDA / 节律 / 下界"节点覆盖，单独画 pipeline 会重复，故省略。
- **指标体系图**：本题为优化类，无主观赋权评价子问题，不适用。
- **决策树/规则图**：未涉及分类或规则推断。

## 导出与自检记录

| 项 | 备注 |
| --- | --- |
| DrawIO CLI 检查 | `command -v drawio`、`draw.io`、`draw.io.exe` 均未找到。`C:/Program Files{,(x86)}`、`%LOCALAPPDATA%/Programs/` 也无安装。 |
| 备用方案 | 使用 matplotlib 直接绘制等价 PDF 流程图（`code/fig_roadmap.py`、`fig_flow_q{1,2,3}_pdf.py`、`fig_model_pdf.py`），保证论文可直接引用。 |
| 源文件保留 | 同名 `.drawio` 文件保留为可编辑源，方便后期在 [diagrams.net](https://app.diagrams.net) 在线编辑或本地导出。 |
| PDF 自检 | 5 个 PDF 文件均已生成，节点无重叠，箭头方向清晰，文字为中文（与论文语言一致），样式分色：决策=红 / 处理=黄 / 数据=绿 / 终止=紫 / 起始=蓝。 |
| 无 Unicode 渲染问题 | 替换了 ⌈ ⌉、∀ 等无字形字符为 ASCII 描述（如 `ceil(...)`、"每个 H"）。 |

如需手动导出 PDF（在已安装 DrawIO 的机器上）：

```bash
drawio --export --format pdf --crop --output figures/fig_roadmap.pdf figures/fig_roadmap.drawio
# 同理 fig_flow_q1/q2/q3 与 fig_model
```

## 给论文阶段的嵌入建议

| 图 | 建议章节 | 建议 caption |
| --- | --- | --- |
| `fig_roadmap.pdf` | 问题重述与分析（§2 末） | 技术路线图：从原始进货量数据到三问求解与论文产出的方法链 |
| `fig_model.pdf` | 模型建立（§4 开头） | 模型变量与约束关系：决策变量 (δ, ξ, y) 经派生量 (W, Cap) 进入约束 |
| `fig_flow_q1.pdf` | 问题一求解（§5） | 问题一求解流程：枚举起点 + 5 变量 LP + 双级目标（Z 最小 / peak 最低） |
| `fig_flow_q2.pdf` | 问题二求解（§6） | 问题二求解流程：MILP（连续 y）+ 累计可行性 + 16 点截止校验 |
| `fig_flow_q3.pdf` | 问题三求解（§7） | 问题三求解流程：模式生成 + LP 松弛 + MILP 列精化 |

具体的 Typst `#figure(image("../../figures/xxx.pdf", width: 85%), caption: [...])` 代码由 `5writing` 阶段写入。
