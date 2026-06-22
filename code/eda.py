"""T4: EDA 图表 F-EDA-1/2/3.pdf 与摘要 CSV。"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    FIGURES_DIR, RESULTS_DIR, C, L,
    load_volume, lower_bound_per_day, setup_chinese_font, save_pdf,
)


def main() -> None:
    setup_chinese_font()
    v = load_volume()  # (30, 24)
    days = np.arange(1, 31)
    hours = np.arange(24)

    # --- F-EDA-1: 30×24 热力图 ---
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(v, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("小时")
    ax.set_ylabel("日序")
    ax.set_xticks(hours)
    ax.set_yticks(np.arange(0, 30, 2))
    ax.set_yticklabels(days[::2])
    cb = fig.colorbar(im, ax=ax, label="进货量 (件)")
    save_pdf(fig, "F-EDA-1_heatmap.pdf")
    plt.close(fig)

    # --- F-EDA-2: 小时进货量均值与极差 ---
    means = v.mean(axis=0)
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(hours, mins, maxs, alpha=0.3, label="30 日极差范围")
    ax.plot(hours, means, marker="o", color="C3", label="均值")
    ax.set_xlabel("小时")
    ax.set_ylabel("进货量 (件)")
    ax.set_xticks(hours)
    ax.grid(alpha=0.3)
    ax.legend()
    save_pdf(fig, "F-EDA-2_hourly_profile.pdf")
    plt.close(fig)

    # --- F-EDA-3: 每日总进货量柱状图 + 下界 ---
    totals = v.sum(axis=1)
    lb = lower_bound_per_day(v)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.bar(days, totals, color="C0", alpha=0.85)
    ax1.set_ylabel("日总进货量 (件)")
    ax1.grid(alpha=0.3)
    ax1.axhline(totals.mean(), color="C3", ls="--", label=f"均值 {totals.mean():.0f}")
    ax1.legend()
    ax2.bar(days, lb, color="C2", alpha=0.85)
    ax2.set_ylabel("人数下界 $L_d$")
    ax2.set_xlabel("日序")
    ax2.grid(alpha=0.3)
    ax2.set_xticks(days[::2])
    save_pdf(fig, "F-EDA-3_daily_total_and_LB.pdf")
    plt.close(fig)

    # --- 统计摘要 CSV ---
    summary = {
        "total": int(v.sum()),
        "daily_total_min": int(totals.min()),
        "daily_total_max": int(totals.max()),
        "daily_total_mean": float(totals.mean()),
        "daily_total_min_day": int(np.argmin(totals)) + 1,
        "daily_total_max_day": int(np.argmax(totals)) + 1,
        "hour_max_global": int(v.max()),
        "hour_min_global": int(v.min()),
        "morning_0_11_avg": float(v[:, :12].sum(axis=1).mean()),
        "morning_share": float(v[:, :12].sum() / v.sum()),
        "hourly_mean": means.tolist(),
        "hourly_max": maxs.tolist(),
        "hourly_min": mins.tolist(),
        "lower_bound_per_day": lb.tolist(),
        "lower_bound_max": int(lb.max()),
        "lower_bound_min": int(lb.min()),
        "lower_bound_mean": float(lb.mean()),
        "lower_bound_total": int(lb.sum()),
    }
    out = RESULTS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("EDA done. Wrote", out)
    print("Figures:", *FIGURES_DIR.glob("F-EDA-*.pdf"), sep="\n  ")


if __name__ == "__main__":
    main()
