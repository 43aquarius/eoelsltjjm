"""T1: 问题一——枚举 C(24,5)=42504 个起点组合 + 每个组合解 5 变量 LP。

单日：固定 5 起点 S=(s1..s5)，求
    min sum xi[k]
    s.t. sum_{h<=H, k} 25*A[s_k,h]*xi[k] >= cumv[H], ∀H
         xi[k] >= 0  (LP 松弛，事后取整)

枚举所有 42504 组合，选 Z 最小的。LP 用 scipy.linprog。
最终 xi 取整：因 b 整数，LP 最优在分数顶点，向上取整后验证 + 调整。
"""

from __future__ import annotations

import json
import time
from itertools import accumulate, combinations
from math import ceil

import numpy as np
from scipy.optimize import linprog

from utils import C, N_DAYS, N_HOURS, N_SHIFT, RESULTS_DIR, build_A, load_volume, lower_bound_per_day


def precompute_M(A: np.ndarray) -> np.ndarray:
    """M[H, s] = sum_{h<=H} A[s,h]  -> 每起点对累计 H 的小时贡献数。"""
    M = np.zeros((N_HOURS, N_HOURS), dtype=int)
    for s in range(N_HOURS):
        cum = 0
        for h in range(N_HOURS):
            cum += A[s, h]
            M[h, s] = cum
    return M


def solve_combo_lp(M_sub: np.ndarray, cumv: np.ndarray) -> tuple[float, np.ndarray]:
    """对单一 5 起点组合，解 LP min sum xi  s.t. C * M_sub @ xi >= cumv, xi >= 0."""
    # scipy: min c^T x  s.t. A_ub x <= b_ub, x >= 0
    # 我们要 C * M_sub @ xi >= cumv -> -C * M_sub @ xi <= -cumv
    c = np.ones(N_SHIFT)
    A_ub = -C * M_sub.astype(float)  # (24, 5)
    b_ub = -cumv.astype(float)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * N_SHIFT,
                  method="highs")
    if not res.success:
        return float("inf"), np.zeros(N_SHIFT)
    return res.fun, res.x


def integerize(xi_float: np.ndarray, M_sub: np.ndarray, cumv: np.ndarray) -> np.ndarray:
    """LP 解整数化：先向上取整，逐 H 调整。"""
    xi = np.ceil(xi_float - 1e-9).astype(int)
    # 验证 + 增量补救
    for H in range(N_HOURS):
        prod = C * (M_sub[H] @ xi)
        if prod < cumv[H]:
            need = ceil((cumv[H] - prod) / C)
            # 加到对累计 H 贡献最大的起点
            best = int(np.argmax(M_sub[H]))
            xi[best] += need
    return xi


def peak_of(starts: list[int], xi: np.ndarray, A: np.ndarray) -> int:
    """对 5 元起点 + 5 元 xi, 返回 max_h W_h。"""
    W = np.zeros(N_HOURS, dtype=int)
    for k, s in enumerate(starts):
        for h in range(N_HOURS):
            if A[s, h]:
                W[h] += int(xi[k])
    return int(W.max())


def solve_one_day(v_d: np.ndarray, A: np.ndarray, M_pre: np.ndarray | None = None,
                  early_stop_lb: int | None = None) -> dict:
    """两级目标：(1) min sum xi ; (2) among optimal Z, min max_h W_h."""
    if M_pre is None:
        M_pre = precompute_M(A)
    cumv = np.array(list(accumulate(int(v_d[h]) for h in range(N_HOURS))), dtype=int)
    best_Z = float("inf")
    best_peak = float("inf")
    best_starts = None
    best_xi_int = None
    t0 = time.time()
    for combo in combinations(range(N_HOURS), N_SHIFT):
        M_sub = M_pre[:, list(combo)]
        Z_lp, x_lp = solve_combo_lp(M_sub, cumv)
        if Z_lp > best_Z:  # LP lower bound exceeds current incumbent
            continue
        x_int = integerize(x_lp, M_sub, cumv)
        Z_int = int(x_int.sum())
        if Z_int > best_Z:
            continue
        peak = peak_of(list(combo), x_int, A)
        if (Z_int < best_Z) or (Z_int == best_Z and peak < best_peak):
            best_Z = Z_int
            best_peak = peak
            best_starts = combo
            best_xi_int = x_int
    elapsed = time.time() - t0
    # 还原全量 xi (24,) 数组
    xi_full = np.zeros(N_HOURS, dtype=int)
    for k, s in enumerate(best_starts):
        xi_full[s] = int(best_xi_int[k])
    return {
        "status": "Optimal",
        "elapsed_s": round(elapsed, 2),
        "starts": list(best_starts),
        "xi": xi_full.tolist(),
        "n_per_shift": [int(best_xi_int[k]) for k in range(N_SHIFT)],
        "Z": best_Z,
        "peak_hour_workers": best_peak,
    }


def verify(v: np.ndarray, xi_arr: np.ndarray) -> dict:
    A = build_A()
    cap = (xi_arr @ A) * C
    cumcap = cap.cumsum(axis=1)
    cumv = v.cumsum(axis=1)
    slack = cumcap - cumv
    return {
        "min_cum_slack": int(slack.min()),
        "feasible_cum": bool((slack >= 0).all()),
        "daily_close": bool((cumcap[:, 23] >= cumv[:, 23]).all()),
        "binding_cum_count": int((slack == 0).sum()),
    }


def _solve_one_day_wrapper(args):
    d, v_d, A, M_pre = args
    return d, solve_one_day(v_d, A, M_pre)


def main() -> None:
    from multiprocessing import Pool

    v = load_volume()
    lb = lower_bound_per_day(v)
    A = build_A()
    M_pre = precompute_M(A)
    print(f"LB total = {int(lb.sum())}", flush=True)

    print("Solving Q1 (enumeration, 30 days, parallel)...", flush=True)
    per_day = [None] * N_DAYS
    xi_arr = np.zeros((N_DAYS, N_HOURS), dtype=int)
    t0 = time.time()
    args_list = [(d, v[d], A, M_pre) for d in range(N_DAYS)]
    with Pool(processes=8) as pool:
        for d, r in pool.imap_unordered(_solve_one_day_wrapper, args_list):
            per_day[d] = r
            xi_arr[d] = np.array(r["xi"], dtype=int)
            print(f"  day {d+1:2d}: Z={r['Z']:4d} peak={r['peak_hour_workers']:3d} starts={r['starts']} ({r['elapsed_s']}s)", flush=True)
    elapsed = time.time() - t0
    Z_total = int(xi_arr.sum())
    print(f"Q1 total = {Z_total}  in {elapsed:.1f}s", flush=True)

    vstat = verify(v, xi_arr)
    print("Verify:", vstat, flush=True)

    out = {
        "mode": "enumeration_C24_5",
        "Z_total": Z_total,
        "Z_per_day": [r["Z"] for r in per_day],
        "starts_per_day": [r["starts"] for r in per_day],
        "n_per_shift_per_day": [r["n_per_shift"] for r in per_day],
        "xi_per_day_per_start": xi_arr.tolist(),
        "status_per_day": [r["status"] for r in per_day],
        "verify": vstat,
        "lower_bound_per_day": lb.tolist(),
        "lower_bound_total": int(lb.sum()),
        "gap_to_lb_pct": round((Z_total - int(lb.sum())) / int(lb.sum()) * 100, 2),
        "elapsed_total_s": round(elapsed, 1),
    }
    (RESULTS_DIR / "q1_solution.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved q1_solution.json", flush=True)


if __name__ == "__main__":
    main()
