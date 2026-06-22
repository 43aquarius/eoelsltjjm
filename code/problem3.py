"""T3: 问题三——月度最少招工 N。

输入：Q2 输出 D_d 与每日 5 班次需求 n_{d,k}。

模型用 "出勤模式" (pattern) 法：
- 每个 pattern p 是长度 30 的 0/1 向量，sum=23，无 8 连续 1（≤7 连工）。
- 决策：n_p ∈ ℤ_+ 表示采用模式 p 的工人数。
- 总人数 N = sum_p n_p。
- 约束：sum_p n_p * p[d] >= D_d, ∀d。
- 目标：min N。

可行 pattern 总数较小（C(30,23) ≈ 2M 中过滤后约几十万），但为效率仅生成 cyclic 旋转 + 局部变换的高质量子集。

第二阶段：把每日出勤的 sum_i w[i,d] 名工人贪心分到 5 个班次满足 n_{d,k}。
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from math import ceil

import numpy as np
import pulp

from utils import N_DAYS, RESULTS_DIR


def generate_patterns(max_consec: int = 7, n_off: int = 7) -> list[tuple[int, ...]]:
    """枚举所有 30 位、含 n_off 个 0、最长连续 1 不超过 max_consec 的模式。"""
    patterns = []
    for offs in combinations(range(N_DAYS), n_off):
        # 检验最长连续工作天数
        w = [1] * N_DAYS
        for o in offs:
            w[o] = 0
        run = 0; max_run = 0
        for x in w:
            run = run + 1 if x else 0
            if run > max_run:
                max_run = run
        if max_run <= max_consec:
            patterns.append(tuple(w))
    return patterns


def solve(D: np.ndarray, time_limit: int = 120) -> dict:
    print(f"D total = {int(D.sum())}, max = {int(D.max())}", flush=True)
    N_LB = max(int(ceil(D.sum() / 23)), int(D.max()))
    print(f"N_LB = {N_LB}", flush=True)

    print("Generating patterns ...", flush=True)
    t0 = time.time()
    patterns = generate_patterns()
    print(f"  {len(patterns)} patterns in {time.time()-t0:.1f}s", flush=True)

    P = np.array(patterns, dtype=np.int8)  # (n_patterns, 30)

    # LP relaxation via scipy.linprog: min c^T n  s.t. -P^T n <= -D, n >= 0
    print("Solving LP relaxation via scipy.linprog (HiGHS) ...", flush=True)
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    n_p = len(patterns)
    c = np.ones(n_p)
    A_ub = -csr_matrix(P.T.astype(float))  # (30, n_p)
    b_ub = -D.astype(float)
    t0 = time.time()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n_p, method="highs")
    elapsed = time.time() - t0
    print(f"  LP status: {res.status} {res.message} in {elapsed:.1f}s; LP obj = {res.fun:.2f}", flush=True)
    assert res.success, "LP failed"

    # Round LP fractional n_p^* up to nearest integer
    n_lp = res.x
    # 用 MILP 在 LP 活跃的 pattern 子集上精化整数解（列生成 1 轮）
    active_p = np.where(n_lp > 1e-6)[0]
    print(f"  active patterns: {len(active_p)}; refining with MILP ...", flush=True)
    P_act = P[active_p]  # (n_act, 30)
    t0 = time.time()
    prob = pulp.LpProblem("Q3_refine", pulp.LpMinimize)
    n_var = [pulp.LpVariable(f"n_{p}", lowBound=0, cat="Integer") for p in range(len(active_p))]
    prob += pulp.lpSum(n_var)
    for d in range(N_DAYS):
        prob += pulp.lpSum(n_var[p] * int(P_act[p, d]) for p in range(len(active_p))) >= int(D[d])
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=120, threads=8)
    status_milp = prob.solve(solver)
    print(f"  MILP refine status: {pulp.LpStatus[status_milp]} in {time.time()-t0:.1f}s", flush=True)
    n_val_act = np.array([int(round(v.value() or 0)) for v in n_var])
    # 扩展回完整 n_val
    n_val = np.zeros(n_p, dtype=int)
    n_val[active_p] = n_val_act
    cov = P.T @ n_val
    while (cov < D).any():
        d_short = int(np.argmin(cov - D))
        cand = np.where(P[:, d_short] == 1)[0]
        n_val[cand[0]] += 1
        cov = P.T @ n_val
    N_total = int(n_val.sum())
    used_patterns = [(patterns[p], int(n_val[p])) for p in range(len(patterns)) if n_val[p] > 0]
    print(f"N = {N_total}  (used {len(used_patterns)} patterns)", flush=True)

    # 展开为 w 矩阵 (N_total × 30)
    w_arr = []
    for pat, count in used_patterns:
        for _ in range(count):
            w_arr.append(list(pat))
    w_arr = np.array(w_arr, dtype=int)
    return {
        "feasible": True,
        "N": N_total,
        "N_LB": N_LB,
        "LP_obj": float(res.fun),
        "status": "LP-relaxed-ceil",
        "elapsed_s": round(elapsed, 1),
        "n_patterns_total": len(patterns),
        "n_patterns_used": len(used_patterns),
        "used_patterns": [(list(p), c) for p, c in used_patterns],
        "w": w_arr,
    }


def stage2_assign_to_shifts(w_arr: np.ndarray, n_per_shift: list[list[int]]) -> np.ndarray:
    """每日把出勤工人分到 5 班次。"""
    N = w_arr.shape[0]
    a = np.zeros((N, N_DAYS, 5), dtype=int)
    for d in range(N_DAYS):
        active = np.where(w_arr[:, d] == 1)[0]
        idx = 0
        for k in range(5):
            need = n_per_shift[d][k] if k < len(n_per_shift[d]) else 0
            for _ in range(need):
                if idx >= len(active):
                    raise RuntimeError(f"Not enough workers day {d+1} shift {k}")
                a[active[idx], d, k] = 1
                idx += 1
    return a


def verify(w_arr: np.ndarray, a_arr: np.ndarray, D: np.ndarray, n_per_shift: list[list[int]]) -> dict:
    N = w_arr.shape[0]
    daily_total = w_arr.sum(axis=0)
    works_per_person = w_arr.sum(axis=1)
    max_consec = []
    for i in range(N):
        run = 0; m = 0
        for d in range(N_DAYS):
            run = run + 1 if w_arr[i, d] else 0
            if run > m: m = run
        max_consec.append(m)
    shift_cover = True
    for d in range(N_DAYS):
        for k in range(5):
            need = n_per_shift[d][k] if k < len(n_per_shift[d]) else 0
            if int(a_arr[:, d, k].sum()) < need:
                shift_cover = False; break
    return {
        "daily_total_ge_D": bool((daily_total >= D).all()),
        "min_daily_total_minus_D": int((daily_total - D).min()),
        "all_work_23": bool((works_per_person == 23).all()),
        "max_consec_work_global": int(max(max_consec)),
        "consec_ok": bool(max(max_consec) <= 7),
        "shift_cover_ok": shift_cover,
        "max_consec_distribution": {f"{k}": int(sum(1 for x in max_consec if x == k)) for k in sorted(set(max_consec))},
    }


def main() -> None:
    q2 = json.loads((RESULTS_DIR / "q2_solution.json").read_text(encoding="utf-8"))
    D = np.array(q2["Z_per_day"], dtype=int)
    n_per_shift = q2["n_per_shift_per_day"]
    r = solve(D, time_limit=180)
    w_arr = r["w"]
    a_arr = stage2_assign_to_shifts(w_arr, n_per_shift)
    vstat = verify(w_arr, a_arr, D, n_per_shift)
    print("Verify:", vstat, flush=True)
    out = {
        "N": int(r["N"]),
        "N_LB": int(r["N_LB"]),
        "stage1_status": r["status"],
        "stage1_elapsed_s": r["elapsed_s"],
        "n_patterns_total": r["n_patterns_total"],
        "n_patterns_used": r["n_patterns_used"],
        "used_patterns": r["used_patterns"],
        "D_per_day": D.tolist(),
        "w_per_worker_per_day": w_arr.tolist(),
        "a_per_worker_per_day_per_shift": a_arr.tolist(),
        "verify": vstat,
    }
    (RESULTS_DIR / "q3_solution.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved q3_solution.json", flush=True)


if __name__ == "__main__":
    main()
