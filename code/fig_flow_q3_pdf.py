"""问题三求解流程 PDF。"""
import matplotlib.pyplot as plt
from utils import FIGURES_DIR, setup_chinese_font
from fig_drawio_pdf import (NODE_STYLE_PROC, NODE_STYLE_DATA, NODE_STYLE_DEC,
                             NODE_STYLE_END, NODE_STYLE_START, node, arrow)

setup_chinese_font()
fig, ax = plt.subplots(figsize=(9, 10))
ax.set_xlim(0, 9); ax.set_ylim(0, 10); ax.axis("off")
ax.text(4.5, 9.6, "问题三求解流程：模式 + LP 松弛 + MILP 精化", ha="center", fontsize=13, fontweight="bold")

steps = [
    (4.5, 9.0, "开始", NODE_STYLE_START, ""),
    (4.5, 8.2, "读取 Q2 输出 D$_d$ 与 n$_{d,k}$", NODE_STYLE_DATA, ""),
    (4.5, 7.4, "下界 N$_{LB}$ = max(ceil(∑D/23), max D$_d$) = 581", NODE_STYLE_DATA, ""),
    (4.5, 6.5, "枚举合法 30 日模式\nsum=23, 最长连续 1 ≤7", NODE_STYLE_PROC, "767,544 条"),
    (4.5, 5.4, "LP 松弛 (scipy.linprog HiGHS)\nmin ∑n$_p$ s.t. P$^T$n ≥ D, n≥0", NODE_STYLE_PROC, "LP 最优 = 581.00"),
    (4.5, 4.3, "筛 LP 活跃模式 (n$_p$ > 0)", NODE_STYLE_END, "14 条活跃"),
    (4.5, 3.4, "MILP 精化 (CBC)\nn$_p$ ∈ Z$_+$ 限定在活跃模式", NODE_STYLE_DEC, ""),
    (4.5, 2.4, "N = ∑n$_p$；展开 w$_{i,d}$（N×30）", NODE_STYLE_DATA, ""),
    (4.5, 1.6, "第二阶段：每日 D$_d$ 工人贪心分到 5 班次", NODE_STYLE_DATA, ""),
    (4.5, 0.8, "校验：23 天 / ≤7 连工 / 班次覆盖", NODE_STYLE_DATA, ""),
    (4.5, 0.05, "导出 q3_solution.json (N=581)", NODE_STYLE_END, ""),
]
for x, y, text, style, _ in steps:
    node(ax, x, y, 0, 0, text, style, fontsize=9)

# annotations
for x, y, _, _, anno in steps:
    if anno:
        ax.text(7.2, y, anno, fontsize=9, color="#666", style="italic")

for i in range(len(steps) - 1):
    arrow(ax, steps[i][0], steps[i][1] - 0.25, steps[i+1][0], steps[i+1][1] + 0.25)

fig.savefig(FIGURES_DIR / "fig_flow_q3.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
print("fig_flow_q3.pdf done")
