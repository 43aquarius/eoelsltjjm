"""T2: 问题二每日独立 MILP——24 起点选 5 + 班内 1 小时低产能 (10 件/h) + 16 点截止。

单日模型：
    min  sum_s xi_s
    s.t. (D2) sum_{h<=H} Cap_h >= sum_{h<=H} v_h, ∀H ∈ [0..23]
         (覆盖 D3、D4)
         xi_s <= M * delta_s,  sum delta <= 5
         sum_{j ∈ Shift(s)} y[s,j] = xi_s
         y[s,j] >= 0 整数, j ∈ Shift(s)
    其中 Cap_h = 25 * sum_s A[s,h]*xi_s - 15 * sum_s y[s,h] (h ∈ Shift(s))
"""

from __future__ import annotations

import json
import time
from itertools import accumulate

import numpy as np
import pulp

from utils import (
    C, C_LOW, L, N_DAYS, N_HOURS, N_SHIFT, RESULTS_DIR,
    build_A, load_volume, lower_bound_per_day, shift_hours,
)


def solve_one_day(v_d: np.ndarray, A: np.ndarray, time_limit: int = 60) -> dict:
    shift_idx = {s: shift_hours(s) for s in range(N_HOURS)}
    M = int(np.ceil(v_d.max() / C_LOW)) + 10

    prob = pulp.LpProblem("Q2_day", pulp.LpMinimize)
    xi = {s: pulp.LpVariable(f"xi_{s}", lowBound=0, cat="Integer") for s in range(N_HOURS)}
    delta = {s: pulp.LpVariable(f"delta_{s}", cat="Binary") for s in range(N_HOURS)}
    # y 放宽为连续：xi 为整数 + sum_h y[s,h] = xi[s] 保证总人数整数；
    # y 是"哪小时是低产能"的分配细节，连续解可被后处理为整数排班。
    y = {(s, h): pulp.LpVariable(f"y_{s}_{h}", lowBound=0, cat="Continuous")
         for s in range(N_HOURS) for h in shift_idx[s]}

    prob += pulp.lpSum(xi.values())

    for s in range(N_HOURS):
        prob += xi[s] <= M * delta[s]
    prob += pulp.lpSum(delta.values()) == N_SHIFT
    for s in range(N_HOURS):
        prob += pulp.lpSum(y[(s, h)] for h in shift_idx[s]) == xi[s]

    def cap_expr(h):
        return (C * pulp.lpSum(A[s, h] * xi[s] for s in range(N_HOURS))
                - (C - C_LOW) * pulp.lpSum(y[(s, h)] for s in range(N_HOURS) if h in shift_idx[s]))

    cumv = list(accumulate(int(v_d[h]) for h in range(N_HOURS)))
    for H in range(N_HOURS):
        prob += pulp.lpSum(cap_expr(h) for h in range(H + 1)) >= cumv[H]

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit, threads=2)
    t0 = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0

    xi_v = [int(round(xi[s].value() or 0)) for s in range(N_HOURS)]
    y_v = np.zeros((N_HOURS, N_HOURS), dtype=float)
    for (s, h), var in y.items():
        y_v[s, h] = float(var.value() or 0)
    d_v = [int(round(delta[s].value() or 0)) for s in range(N_HOURS)]
    starts = sorted([s for s, x in enumerate(d_v) if x == 1])
    n_per_shift = [xi_v[s] for s in starts]
    return {
        "status": pulp.LpStatus[status],
        "elapsed_s": round(elapsed, 2),
        "starts": starts,
        "xi": xi_v,
        "y": y_v.tolist(),
        "n_per_shift": n_per_shift,
        "Z": sum(xi_v),
    }


def verify(v: np.ndarray, xi_arr: np.ndarray, y_arr: np.ndarray) -> dict:
    A = build_A()
    W = xi_arr @ A
    Y = y_arr.sum(axis=1)  # over s -> (30, 24)
    cap = C * W - (C - C_LOW) * Y
    cumcap = cap.cumsum(axis=1)
    cumv = v.cumsum(axis=1)
    slack = cumcap - cumv
    return {
        "min_cum_slack": int(slack.min()),
        "feasible_D2": bool((slack >= 0).all()),
        "deadline_D3_ok": bool((cumcap[:, 15] >= cumv[:, 11]).all()),
        "daily_close_D4": bool((cumcap[:, 23] >= cumv[:, 23]).all()),
        "binding_D2_count": int((slack == 0).sum()),
        "min_D3_slack": int((cumcap[:, 15] - cumv[:, 11]).min()),
    }


def main() -> None:
    v = load_volume()
    lb = lower_bound_per_day(v)
    A = build_A()
    print(f"LB total = {int(lb.sum())}")

    print("Solving Q2 per-day MILP (30 days) ...")
    per_day = []
    Z_d = []
    starts_per_day = []
    xi_arr = np.zeros((N_DAYS, N_HOURS), dtype=int)
    y_arr = np.zeros((N_DAYS, N_HOURS, N_HOURS), dtype=int)
    t0 = time.time()
    for d in range(N_DAYS):
        r = solve_one_day(v[d], A)
        per_day.append(r)
        Z_d.append(r["Z"])
        starts_per_day.append(r["starts"])
        for s in range(N_HOURS):
            xi_arr[d, s] = r["xi"][s]
        y_arr[d] = np.array(r["y"], dtype=int)
        print(f"  day {d+1:2d}: Z={r['Z']:4d}  starts={r['starts']}  ({r['status']}, {r['elapsed_s']}s)")
    elapsed = time.time() - t0
    Z_total = sum(Z_d)
    print(f"Total Q2: {Z_total}  in {elapsed:.1f}s")

    vstat = verify(v, xi_arr, y_arr)
    print("Verify:", vstat)

    # vs Q1
    q1 = json.loads((RESULTS_DIR / "q1_solution.json").read_text(encoding="utf-8"))
    Z1 = np.array(q1["Z_per_day"])
    Z2 = np.array(Z_d)
    print(f"vs Q1: Z1={Z1.sum()}, Z2={Z2.sum()}, delta={Z2.sum()-Z1.sum()} ({(Z2.sum()/Z1.sum()-1)*100:.2f}%)")
    print(f"Z2 >= Z1 per-day? {bool((Z2 >= Z1).all())}")

    out = {
        "mode": "per_day_cumulative",
        "Z_total": int(Z_total),
        "Z_per_day": Z_d,
        "starts_per_day": starts_per_day,
        "n_per_shift_per_day": [r["n_per_shift"] for r in per_day],
        "xi_per_day_per_start": xi_arr.tolist(),
        "y_per_day_per_start_per_hour": y_arr.tolist(),
        "status_per_day": [r["status"] for r in per_day],
        "verify": vstat,
        "compare_q1": {"Z1": int(Z1.sum()), "Z2": int(Z2.sum()),
                       "delta": int(Z2.sum() - Z1.sum()),
                       "Z2_ge_Z1_each_day": [bool(x) for x in (Z2 >= Z1)]},
        "lower_bound_total": int(lb.sum()),
    }
    path = RESULTS_DIR / "q2_solution.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main()
