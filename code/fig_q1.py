"""T5: Q1 数据图。"""

from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import (C, FIGURES_DIR, RESULTS_DIR, N_DAYS, N_HOURS,
                   build_A, load_volume, setup_chinese_font, save_pdf)


def main() -> None:
    setup_chinese_font()
    v = load_volume()
    sol = json.loads((RESULTS_DIR / "q1_solution.json").read_text(encoding="utf-8"))
    xi = np.array(sol["xi_per_day_per_start"])
    Z_d = np.array(sol["Z_per_day"])
    starts_per_day = sol["starts_per_day"]
    lb = np.array(sol["lower_bound_per_day"])

    days = np.arange(1, 31)

    # F-Q1-1: 每日总工人数 + 下界
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(days, Z_d, color="C0", alpha=0.85, label="Q1 每日工人数 $Z_d$")
    ax.plot(days, lb, "C3o-", label="下界 $L_d$", markersize=5)
    ax.set_xlabel("日序")
    ax.set_ylabel("工人数")
    ax.set_xticks(days[::2])
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    save_pdf(fig, "F-Q1-1_daily_workers.pdf")
    plt.close(fig)

    # F-Q1-2: 3 个典型日（min/median/max Z）小时人力 vs 需求
    A = build_A()
    per_hour = xi @ A  # (30, 24)
    cap = per_hour * C
    Z_sorted = np.argsort(Z_d)
    typical = [("min", Z_sorted[0]), ("中位", Z_sorted[len(Z_sorted)//2]), ("max", Z_sorted[-1])]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    hours = np.arange(24)
    for ax, (label, d) in zip(axes, typical):
        ax.bar(hours, v[d], color="C0", alpha=0.5, label="进货量 $v_h$")
        ax.plot(hours, cap[d], "C3-o", label="有效产能 $25 W_h$")
        ax.set_ylabel(f"d={d+1} ({label}, $Z={Z_d[d]}$)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("小时")
    axes[-1].set_xticks(hours)
    save_pdf(fig, "F-Q1-2_typical_hourly.pdf")
    plt.close(fig)

    # F-Q1-3: 起点选择热度
    starts_count = np.zeros(N_HOURS, dtype=int)
    for sd in starts_per_day:
        for s in sd:
            starts_count[s] += 1
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(np.arange(N_HOURS), starts_count, color="C2", alpha=0.85)
    ax.set_xlabel("班次起点小时")
    ax.set_ylabel("被选用天数")
    ax.set_xticks(np.arange(N_HOURS))
    ax.grid(alpha=0.3, axis="y")
    save_pdf(fig, "F-Q1-3_start_frequency.pdf")
    plt.close(fig)

    print("Q1 figures done.")


if __name__ == "__main__":
    main()
