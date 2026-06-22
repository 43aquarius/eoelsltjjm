"""问题一求解流程 PDF。"""
import matplotlib.pyplot as plt
from utils import FIGURES_DIR, setup_chinese_font
from fig_drawio_pdf import (NODE_STYLE_PROC, NODE_STYLE_DATA, NODE_STYLE_DEC,
                             NODE_STYLE_END, NODE_STYLE_START, node, arrow)

setup_chinese_font()
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 8); ax.set_ylim(0, 11); ax.axis("off")
ax.text(4, 10.6, "问题一求解流程：每日独立 + 双级目标", ha="center", fontsize=13, fontweight="bold")

steps = [
    (4, 10.0, "开始", NODE_STYLE_START),
    (4, 9.2, "读取 v$_{d,h}$，构造覆盖矩阵 A$_{s,h}$（24×24）", NODE_STYLE_DATA),
    (4, 8.3, "对每日 d：计算累计 cumv$_H$ 与下界 L$_d$", NODE_STYLE_DATA),
    (4, 7.4, "枚举 C(24,5)=42504 个起点组合 S", NODE_STYLE_PROC),
    (4, 6.5, "对每个 S 解 5 变量 LP\nmin ∑ξ$_k$ s.t. C·M$_S$·ξ ≥ cumv", NODE_STYLE_PROC),
    (4, 5.4, "Z$_{LP}$ < 当前最优？", NODE_STYLE_DEC),
    (4, 4.3, "向上取整 + 补救 ξ\n计算 peak = max$_h$ W$_h$\n按 (Z, peak) 字典序更新最优", NODE_STYLE_END),
    (4, 3.0, "所有组合枚举完毕？", NODE_STYLE_DEC),
    (4, 1.9, "保存 ξ$_{d,·}$, starts$_d$；d←d+1", NODE_STYLE_DATA),
    (4, 1.0, "所有日已处理？", NODE_STYLE_DEC),
    (4, 0.2, "导出 q1_solution.json + 校验 ≥ L", NODE_STYLE_END),
]
for x, y, text, style in steps:
    node(ax, x, y, 0, 0, text, style, fontsize=9)

for i in range(len(steps) - 1):
    arrow(ax, steps[i][0], steps[i][1] - 0.25, steps[i+1][0], steps[i+1][1] + 0.25)

# 是/否 labels
ax.text(4.1, 5.0, "是", fontsize=9, color="#555")
ax.text(5.5, 5.3, "否", fontsize=9, color="#555")
# back-loop arrows
ax.annotate("", xy=(6.5, 6.5), xytext=(6.5, 3.0),
            arrowprops=dict(arrowstyle="->", color="#999", lw=1.0, connectionstyle="arc3,rad=0.2"))
ax.text(6.7, 4.7, "否", fontsize=9, color="#555")
ax.annotate("", xy=(0.6, 8.3), xytext=(0.6, 1.0),
            arrowprops=dict(arrowstyle="->", color="#999", lw=1.0))
ax.text(0.3, 4.7, "否", fontsize=9, color="#555")

fig.savefig(FIGURES_DIR / "fig_flow_q1.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
print("fig_flow_q1.pdf done")
