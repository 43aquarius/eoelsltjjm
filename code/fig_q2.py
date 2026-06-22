"""T6: Q2 数据图。"""

from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import (C, C_LOW, FIGURES_DIR, RESULTS_DIR, N_DAYS, N_HOURS,
                   build_A, load_volume, setup_chinese_font, save_pdf)


def main() -> None:
    setup_chinese_font()
    v = load_volume()
    q2 = json.loads((RESULTS_DIR / "q2_solution.json").read_text(encoding="utf-8"))
    q1 = json.loads((RESULTS_DIR / "q1_solution.json").read_text(encoding="utf-8"))
    xi = np.array(q2["xi_per_day_per_start"])
    y = np.array(q2["y_per_day_per_start_per_hour"])
    Z2 = np.array(q2["Z_per_day"])
    Z1 = np.array(q1["Z_per_day"])
    days = np.arange(1, 31)

    A = build_A()
    W = xi @ A
    Y = y.sum(axis=1)
    cap = C * W - (C - C_LOW) * Y
    cumcap = cap.cumsum(axis=1)
    cumv = v.cumsum(axis=1)

    # F-Q2-1: 累计进货 vs 累计产能 + 16 点垂线（3 个典型日）
    Z_sorted = np.argsort(Z2)
    typical = [("min", Z_sorted[0]), ("中位", Z_sorted[len(Z_sorted)//2]), ("max", Z_sorted[-1])]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    hours = np.arange(24)
    for ax, (label, d) in zip(axes, typical):
        ax.plot(hours, cumv[d], "C0-o", label="累计进货 $\\sum v$")
        ax.plot(hours, cumcap[d], "C3-s", label="累计产能 $\\sum 25W-15Y$")
        ax.axvline(15.5, color="gray", ls=":", label="16:00 截止")
        ax.set_ylabel(f"d={d+1} ({label}, $Z={Z2[d]}$)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("小时")
    axes[-1].set_xticks(hours)
    save_pdf(fig, "F-Q2-1_cumulative_deadline.pdf")
    plt.close(fig)

    # F-Q2-2: Q2 vs Q1 每日人数对比
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    ax.bar(days - width/2, Z1, width, color="C0", alpha=0.85, label="Q1 $Z^{(1)}_d$")
    ax.bar(days + width/2, Z2, width, color="C3", alpha=0.85, label="Q2 $Z^{(2)}_d$")
    ax.set_xlabel("日序")
    ax.set_ylabel("工人数")
    ax.set_xticks(days[::2])
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    save_pdf(fig, "F-Q2-2_Q2_vs_Q1.pdf")
    plt.close(fig)

    # F-Q2-3: 低产能小时分布（按起点）— 显示 y 在 24 小时的分布
    fig, ax = plt.subplots(figsize=(9, 3.5))
    total_low = Y.sum(axis=0)  # (24,) sum over days
    ax.bar(np.arange(N_HOURS), total_low, color="C4", alpha=0.85)
    ax.set_xlabel("小时")
    ax.set_ylabel("低产能工人 (累计 30 日)")
    ax.set_xticks(np.arange(N_HOURS))
    ax.grid(alpha=0.3, axis="y")
    save_pdf(fig, "F-Q2-3_low_cap_hour_distribution.pdf")
    plt.close(fig)
    print("Q2 figures done.")


if __name__ == "__main__":
    main()
