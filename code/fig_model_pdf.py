"""模型变量关系图 PDF。"""
import matplotlib.pyplot as plt
from utils import FIGURES_DIR, setup_chinese_font
from fig_drawio_pdf import (NODE_STYLE_PROC, NODE_STYLE_DATA, NODE_STYLE_DEC,
                             NODE_STYLE_END, NODE_STYLE_START, node, arrow)

setup_chinese_font()
fig, ax = plt.subplots(figsize=(11.5, 7))
ax.set_xlim(0, 11.5); ax.set_ylim(0, 7); ax.axis("off")
ax.text(5.75, 6.6, "模型变量与约束关系", ha="center", fontsize=14, fontweight="bold")

# Inputs (left)
node(ax, 1.0, 4.0, 0, 0, "输入\nv$_{d,h}$", NODE_STYLE_START, 11)
# Decisions (middle-left)
node(ax, 3.0, 5.7, 0, 0, "δ$_s$ ∈ {0,1}\n起点是否选用", NODE_STYLE_PROC, 10)
node(ax, 3.0, 4.0, 0, 0, "ξ$_{d,s}$ ∈ Z$_+$\n每日每起点工人数", NODE_STYLE_PROC, 10)
node(ax, 3.0, 2.3, 0, 0, "y$_{d,s,h}$ ≥ 0\n班内低产能小时", NODE_STYLE_PROC, 10)
# Derived (middle)
node(ax, 5.7, 4.9, 0, 0, "W$_{d,h}$ = ∑$_s$ A$_{s,h}$ξ$_{d,s}$\n小时在岗", NODE_STYLE_DATA, 10)
node(ax, 5.7, 3.0, 0, 0, "Cap$_{d,h}$ = 25·W$_{d,h}$\n   − 15·∑$_s$ y$_{d,s,h}$", NODE_STYLE_DATA, 10)
# Constraints (right)
node(ax, 9.0, 5.9, 0, 0, "∑$_s$ δ$_s$ = 5 (恰 5 起点)", NODE_STYLE_DEC, 10)
node(ax, 9.0, 5.0, 0, 0, "ξ$_{d,s}$ ≤ M·δ$_s$", NODE_STYLE_DEC, 10)
node(ax, 9.0, 3.5, 0, 0, "∑$_{h∈Shift(s)}$ y$_{d,s,h}$ = ξ$_{d,s}$", NODE_STYLE_DEC, 10)
node(ax, 9.0, 2.4, 0, 0, "∑$_{h≤H}$ Cap$_{d,h}$ ≥ ∑$_{h≤H}$ v$_{d,h}$\n(每个 H)", NODE_STYLE_DEC, 10)
node(ax, 9.0, 1.0, 0, 0, "目标\nQ1/Q2: min ∑ξ ; Q3: min N", NODE_STYLE_END, 10)

# arrows
arrow(ax, 1.5, 4.0, 2.4, 4.0)
arrow(ax, 3.0, 5.4, 3.0, 4.4)
arrow(ax, 3.6, 4.0, 5.1, 4.7)
arrow(ax, 3.6, 4.0, 5.1, 3.2)
arrow(ax, 3.6, 2.3, 5.1, 2.7)
arrow(ax, 1.5, 4.0, 5.0, 3.0)
arrow(ax, 6.4, 3.0, 8.4, 2.4)
arrow(ax, 3.6, 5.7, 8.4, 5.9)
arrow(ax, 3.6, 5.7, 8.4, 5.0)
arrow(ax, 3.6, 2.3, 8.4, 3.5)

fig.savefig(FIGURES_DIR / "fig_model.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
print("fig_model.pdf done")
