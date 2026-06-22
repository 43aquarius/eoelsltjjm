"""共享工具：数据加载、班次-小时覆盖矩阵、绘图配置。

供 problem1/2/3 与 EDA 脚本调用。所有路径相对项目根。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import openpyxl
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "附件.xlsx"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

C = 25
C_LOW = 10
L = 8
N_SHIFT = 5
N_DAYS = 30
N_HOURS = 24


def load_volume() -> np.ndarray:
    """读取附件返回 (30, 24) 进货量矩阵 (件)。"""
    wb = openpyxl.load_workbook(DATA_FILE, data_only=True)
    ws = wb["Sheet1"]
    mat = np.zeros((N_DAYS, N_HOURS), dtype=int)
    for row in ws.iter_rows(min_row=2, values_only=True):
        d, h, v = row[0], row[1], row[2]
        mat[d - 1, h] = v
    return mat


def shift_hours(s: int) -> list[int]:
    """以 s 为起点的 8 小时班次覆盖的小时 (mod 24)。"""
    return [(s + i) % N_HOURS for i in range(L)]


def build_A() -> np.ndarray:
    """覆盖矩阵 A[s, h] = 1 ⟺ 起点为 s 的班次覆盖小时 h，shape (24, 24)。"""
    A = np.zeros((N_HOURS, N_HOURS), dtype=int)
    for s in range(N_HOURS):
        for h in shift_hours(s):
            A[s, h] = 1
    return A


def lower_bound_per_day(v: np.ndarray) -> np.ndarray:
    """返回 30 维向量，每日工人数下界 L_d = max(⌈total/200⌉, ⌈max_h/25⌉)。"""
    import math
    total = v.sum(axis=1)
    peak = v.max(axis=1)
    lb = np.maximum(
        np.array([math.ceil(t / (C * L)) for t in total]),
        np.array([math.ceil(p / C) for p in peak]),
    )
    return lb


def setup_chinese_font() -> None:
    """配置 matplotlib 中文字体与 PDF 矢量输出。"""
    for fam in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = [fam]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"


def save_pdf(fig, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, format="pdf")
    return path


if __name__ == "__main__":
    v = load_volume()
    A = build_A()
    print("Loaded v shape:", v.shape, "total =", v.sum())
    assert v.sum() > 2_000_000  # sanity
    print("A row sums (should all be 8):", set(A.sum(axis=1).tolist()))
    print("LB per day:", lower_bound_per_day(v))
