= 问题三的模型建立与求解

== 模型建立

将问题二的输出每日总人数 $D_d = Z^((2))_d$ 视为月度需求向量，工人池规模 $N$ 为待求最少值。每名工人 $i$ 在第 $d$ 日是否出勤记为 $w_(i,d) in {0,1}$，约束如下：
$ &"(E1)" sum_(d=1)^(30) w_(i,d) = 23, &forall i, \
  &"(E2)" sum_(d=t)^(t+7) w_(i,d) lt.eq 7, &forall i, forall t in {1,...,23}, \
  &"(E3)" sum_(i=1)^(N) w_(i,d) gt.eq D_d, &forall d. $

(E1) 强制每人 30 日恰 23 个工作日；(E2) 任意 8 个连续日内工作天数不超过 7 天（即不允许出现 8 个连续工作日）；(E3) 每日总出勤数满足需求。每日的实际班次分配为可行性子问题：在确定 $w_(i,d)$ 后，将出勤者按问题二的 $n_(d,k)$ 顺序分到 5 个班次即可。

== 下界与列式生成法

#strong[理论下界。] 总工时下界 $N gt.eq ceil.l sum_d D_d slash 23 ceil.r = ceil.l 11\,668 slash 23 ceil.r = 508$；峰日下界 $N gt.eq max_d D_d = 581$。综合 $N_"LB" = max(508, 581) = 581$。

#strong[列式生成（pattern）法。] 由于 $N approx 600$、$N times 30 approx 1.8 times 10^4$ 个二元变量，直接 MILP 求解困难。利用工人间对称性，将 30 日"出勤模式"作为决策列：定义 $p$ 为长度 30 的 0/1 向量，$|p|_1 = 23$ 且 $p$ 不含 8 个及以上连续 1。变量 $n_p in bb(Z)_+$ 表示采用模式 $p$ 的工人数，则模型变为
$ min sum_p n_p quad "s.t." quad sum_p p_d dot n_p gt.eq D_d, forall d; quad n_p gt.eq 0. $
求解时先穷举全部合法模式（共 #strong[767\,544] 条），随后两步求解：

+ #strong[LP 松弛]：scipy.linprog (HiGHS) 解 $min sum n_p$ s.t. $P^T n gt.eq D, n gt.eq 0$；得 LP 最优值 581.00；
+ #strong[MILP 精化]：在 LP 取值非零的 14 条活跃模式上用 CBC 求整数解（变量数 14，秒级），最终得 11 条非零模式、$N = 581$。

求解流程见@fig-flow-q3。

#figure(
  image("../../figures/fig_flow_q3.pdf", width: 88%),
  caption: [问题三求解流程],
) <fig-flow-q3>

== 求解结果

#strong[全月最少招工人数 $N = 581$ 人]，与下界完全一致，gap 为 0%。求解耗时：模式生成 6.7 秒、LP 松弛 25.6 秒、MILP 精化 0.1 秒，合计约 35 秒。共使用 11 条非零模式。

@fig-q3-calendar 给出 581 名工人 30 日的出勤矩阵（按出勤模式排序，深色表示出勤）。可以清楚看到 11 类出勤模式形成 11 个水平条带。

#figure(
  image("../../figures/F-Q3-1_worker_calendar.pdf", width: 90%),
  caption: [问题三 581 名工人的 30 日出勤矩阵（按模式排序）],
) <fig-q3-calendar>

@fig-q3-cov 对比每日的出勤总数与需求 $D_d$。所有日子满足 $sum_i w_(i,d) gt.eq D_d$，最小 slack 为 0（峰日精确等于），证明 $N=581$ 是峰值产能受限的紧界。

#figure(
  image("../../figures/F-Q3-2_daily_coverage.pdf", width: 88%),
  caption: [问题三每日出勤总数 $sum_i w_(i,d)$ 与需求 $D_d$ 的对比],
) <fig-q3-cov>

@fig-q3-runs 给出每名工人最长连续工作天数的分布：最长 7 天的有 282 人、6 天的 243 人、5 天的 56 人，全部满足"不超过 7 天连工"的约束。

#figure(
  image("../../figures/F-Q3-3_max_consec_distribution.pdf", width: 75%),
  caption: [问题三每名工人最长连续工作天数分布],
) <fig-q3-runs>

#strong[实际意义。] 581 名工人对应总工时 $581 times 23 = 13\,363$ 工日，而需求为 11\,668 工日，即整月有 1\,695 个"非必要出勤"工日被分摊到非峰日。这是"每人 23 天"硬性约束下的必然冗余，反映工人月度合同与日波动需求之间的张力。

#strong[校验。] (E1) 每人恰 23 天出勤；(E2) 最大连续工作 7 天；(E3) 每日总出勤数满足需求；班次分配的可行性子问题在所有日均成立。
