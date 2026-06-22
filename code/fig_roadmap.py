"""技术路线图 PDF。"""
import matplotlib.pyplot as plt
from utils import FIGURES_DIR, setup_chinese_font
from fig_drawio_pdf import (NODE_STYLE_PROC, NODE_STYLE_DATA, NODE_STYLE_END,
                             NODE_STYLE_START, node, arrow)

setup_chinese_font()
fig, ax = plt.subplots(figsize=(11, 7.5))
ax.set_xlim(0, 11); ax.set_ylim(0, 7.5); ax.axis("off")
ax.text(5.5, 7.1, "物流分拣中心排班——技术路线", ha="center", fontsize=15, fontweight="bold")

# left column
node(ax, 1.4, 6.0, 0, 0, "附件 1\nv$_{d,h}$ (30×24)", NODE_STYLE_START)
node(ax, 1.4, 4.7, 0, 0, "EDA / 节律 / 下界 L$_d$\n日间相关 0.93", NODE_STYLE_DATA)
node(ax, 1.4, 3.2, 0, 0, "假设决断 G1–G5\n整点起点 / 累计口径\n低产能由排班指派", NODE_STYLE_PROC, fontsize=9)

# middle column
node(ax, 5.5, 6.0, 0, 0, "问题一\n每日 MILP（枚举 C(24,5)）\nmin ∑ξ$_s$", NODE_STYLE_DATA, fontsize=10)
node(ax, 5.5, 4.4, 0, 0, "问题二\nQ1 + 16 点截止\n+ 班内低产能 y$_{s,h}$", NODE_STYLE_DATA, fontsize=10)
node(ax, 5.5, 2.8, 0, 0, "问题三\n模式生成 + LP + MILP\nmin N s.t. 23 天 / ≤7 连工", NODE_STYLE_DATA, fontsize=10)

# right column
node(ax, 9.4, 4.4, 0, 0, "结果\nZ$^{(1)}$=10791  Z$^{(2)}$=11668\nN=581 (均达下界)", NODE_STYLE_END, fontsize=10)
node(ax, 9.4, 2.8, 0, 0, "校验\n下界对比 / 单调性 / 约束回代", NODE_STYLE_DATA, fontsize=9)
node(ax, 9.4, 1.2, 0, 0, "Typst 论文撰写", NODE_STYLE_PROC, fontsize=10)

# arrows
arrow(ax, 1.4, 5.7, 1.4, 5.0)
arrow(ax, 1.4, 4.4, 1.4, 3.6)
arrow(ax, 2.5, 3.2, 4.5, 5.8)
arrow(ax, 5.5, 5.5, 5.5, 4.9)
arrow(ax, 5.5, 4.0, 5.5, 3.3)
arrow(ax, 6.6, 4.4, 8.5, 4.4)
arrow(ax, 9.4, 4.0, 9.4, 3.3)
arrow(ax, 9.4, 2.4, 9.4, 1.6)

fig.savefig(FIGURES_DIR / "fig_roadmap.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
print("fig_roadmap.pdf done")
