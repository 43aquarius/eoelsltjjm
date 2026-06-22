"""T7: Q3 数据图。"""

from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import FIGURES_DIR, RESULTS_DIR, N_DAYS, setup_chinese_font, save_pdf


def main() -> None:
    setup_chinese_font()
    q3 = json.loads((RESULTS_DIR / "q3_solution.json").read_text(encoding="utf-8"))
    w = np.array(q3["w_per_worker_per_day"], dtype=int)  # (N, 30)
    D = np.array(q3["D_per_day"])
    N = w.shape[0]
    days = np.arange(1, 31)

    # F-Q3-1: 工人-日 出勤矩阵（按出勤日期模式排序后）
    # 排序：按 pattern hash
    keys = [tuple(w[i]) for i in range(N)]
    order = sorted(range(N), key=lambda i: keys[i])
    w_sorted = w[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(w_sorted, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xlabel("日序")
    ax.set_ylabel("工人编号（按出勤模式排序）")
    ax.set_xticks(np.arange(30)[::2])
    ax.set_xticklabels(days[::2])
    save_pdf(fig, "F-Q3-1_worker_calendar.pdf")
    plt.close(fig)

    # F-Q3-2: 每日出勤总数 vs 需求 D_d
    daily_total = w.sum(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    ax.bar(days - width/2, D, width, color="C0", alpha=0.85, label="$D_d$ (Q2 需求)")
    ax.bar(days + width/2, daily_total, width, color="C2", alpha=0.85, label="出勤总数 $\\sum_i w_{i,d}$")
    ax.set_xlabel("日序")
    ax.set_ylabel("人数")
    ax.set_xticks(days[::2])
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    save_pdf(fig, "F-Q3-2_daily_coverage.pdf")
    plt.close(fig)

    # F-Q3-3: 每人最长连续工作天数分布
    max_consec = []
    for i in range(N):
        run = 0; m = 0
        for d in range(N_DAYS):
            run = run + 1 if w[i, d] else 0
            if run > m: m = run
        max_consec.append(m)
    fig, ax = plt.subplots(figsize=(7, 4))
    vals, counts = np.unique(max_consec, return_counts=True)
    ax.bar(vals, counts, color="C5", alpha=0.85)
    for vi, ci in zip(vals, counts):
        ax.text(vi, ci + N*0.005, str(ci), ha="center")
    ax.set_xlabel("最长连续工作天数")
    ax.set_ylabel("工人数")
    ax.set_xticks(vals)
    ax.grid(alpha=0.3, axis="y")
    save_pdf(fig, "F-Q3-3_max_consec_distribution.pdf")
    plt.close(fig)
    print("Q3 figures done.")


if __name__ == "__main__":
    main()
