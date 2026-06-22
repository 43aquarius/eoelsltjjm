"""问题二求解流程 PDF。"""
import matplotlib.pyplot as plt
from utils import FIGURES_DIR, setup_chinese_font
from fig_drawio_pdf import (NODE_STYLE_PROC, NODE_STYLE_DATA, NODE_STYLE_DEC,
                             NODE_STYLE_END, NODE_STYLE_START, node, arrow)

setup_chinese_font()
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 8); ax.set_ylim(0, 10); ax.axis("off")
ax.text(4, 9.6, "问题二求解流程：MILP + 班内低产能小时", ha="center", fontsize=13, fontweight="bold")

steps = [
    (4, 9.0, "开始", NODE_STYLE_START),
    (4, 8.2, "读取 v$_{d,h}$ 与 Shift(s)（模 24）", NODE_STYLE_DATA),
    (4, 7.4, "对每日 d：构造 PuLP MILP", NODE_STYLE_DATA),
    (4, 6.4, "决策变量\nξ$_s$ ∈ Z$_+$；δ$_s$ ∈ {0,1}\ny$_{s,h}$ ≥ 0 连续（h ∈ Shift(s)）", NODE_STYLE_PROC),
    (4, 5.0, "约束\n∑δ = 5；班内 ∑$_h$ y$_{s,h}$ = ξ$_s$\n累计 Cap$_h$ ≥ 累计 v$_h$（每个 H）", NODE_STYLE_PROC),
    (4, 3.7, "Cap$_h$ = 25·∑$_s$ A$_{s,h}$ξ$_s$\n     − 15·∑$_s$ y$_{s,h}$", NODE_STYLE_END),
    (4, 2.6, "CBC 求解 (threads=2, timeLimit=30s)", NODE_STYLE_DEC),
    (4, 1.6, "校验\nCumCap$_{15}$ ≥ CumArr$_{11}$ (D3)\nZ$^{(2)}_d$ ≥ Z$^{(1)}_d$ 单调性", NODE_STYLE_DATA),
    (4, 0.3, "导出 q2_solution.json", NODE_STYLE_END),
]
for x, y, text, style in steps:
    node(ax, x, y, 0, 0, text, style, fontsize=9)

for i in range(len(steps) - 1):
    arrow(ax, steps[i][0], steps[i][1] - 0.27, steps[i+1][0], steps[i+1][1] + 0.27)

fig.savefig(FIGURES_DIR / "fig_flow_q2.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
print("fig_flow_q2.pdf done")
