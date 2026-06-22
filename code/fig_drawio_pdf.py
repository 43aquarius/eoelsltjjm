"""把 4 张非数据图（roadmap / flow_q1 / flow_q2 / flow_q3 / model）渲染为论文 PDF。

不依赖 drawio CLI，使用 matplotlib 直接绘制等价版本，保证论文可引用。
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from utils import FIGURES_DIR, setup_chinese_font

setup_chinese_font()

NODE_STYLE_PROC = dict(boxstyle="round,pad=0.4", fc="#fff2cc", ec="#d6b656", lw=1.4)
NODE_STYLE_DATA = dict(boxstyle="round,pad=0.4", fc="#d5e8d4", ec="#82b366", lw=1.4)
NODE_STYLE_DEC = dict(boxstyle="round,pad=0.3", fc="#f8cecc", ec="#b85450", lw=1.4)
NODE_STYLE_END = dict(boxstyle="round,pad=0.4", fc="#e1d5e7", ec="#9673a6", lw=1.4)
NODE_STYLE_START = dict(boxstyle="round,pad=0.4", fc="#dae8fc", ec="#6c8ebf", lw=1.4)


def node(ax, x, y, w, h, text, style, fontsize=10):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            bbox=style, wrap=True)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.3))
    if label:
        ax.text((x1 + x2) / 2 + 0.2, (y1 + y2) / 2, label, fontsize=9, color="#555")
