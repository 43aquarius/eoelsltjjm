= 问题一的模型建立与求解

== 模型建立

=== 决策变量与覆盖矩阵

记班次起点集合 $S = {0,1,...,23}$。定义 24 $times$ 24 的小时—起点覆盖矩阵
$ A_(s,h) = cases(1\, & quad (h-s) mod 24 in {0,1,...,7} ,, 0\, & quad "其他".) $

引入二元决策变量 $delta_s in {0,1}$ 表示起点 $s$ 是否被选用，以及非负整数变量 $xi_(d,s) in bb(Z)_+$ 表示第 $d$ 日起点 $s$ 班次的工人数。派生量
$ W_(d,h) = sum_(s=0)^(23) A_(s,h) xi_(d,s) $
即第 $d$ 日第 $h$ 小时的在岗人数。

=== 目标与约束

#strong[主目标：] 全月总工人数最少
$ min space sum_(d=1)^(30) sum_(s=0)^(23) xi_(d,s). $

#strong[二级目标（在主目标取得最优值后）：] 峰值时段在岗人数最小化
$ min space max_(d,h) W_(d,h). $

#strong[约束：]
$ &"(C1)" sum_(s=0)^(23) delta_s = 5 &("每日恰 5 个班次"), \
  &"(C2)" xi_(d,s) lt.eq M dot delta_s, &forall d, s ("链接约束"), \
  &"(C3)" sum_(h=0)^(H) c dot W_(d,h) gt.eq sum_(h=0)^(H) v_(d,h), &forall d in {1,...,30}, forall H in {0,...,23}. $

其中 $c=25$ 件/人·小时，$M$ 为安全大数（取 $ceil.l max_h v_h slash c ceil.r + "余量"$）。约束 (C3) 即"累计可行性"，蕴含 $H=23$ 时的当日清零（H3）。

模型变量与约束的关系如@fig-model 所示。

#figure(
  image("../../figures/fig_model.pdf", width: 95%),
  caption: [模型变量与约束的关系],
) <fig-model>

== 求解算法

由于每日 MILP 含 24 个二元变量 $delta_s$，CBC 直接求解伴随对称性退化导致树搜索缓慢；本文采用#strong["枚举起点组合 + 微型 LP"]方法：
+ 枚举所有 $C(24, 5) = 42504$ 个 5 元起点组合 $S^* = (s_1,...,s_5)$；
+ 对每个 $S^*$，固定 $delta_s$，问题降为只含 5 个连续变量 $xi_k$（$k=1,...,5$）的线性规划；用 scipy.linprog (HiGHS) 解 LP 松弛得 $Z_"LP"$；
+ 若 $Z_"LP" lt.eq$ 当前最优，则将 $xi$ 向上取整并补救为整数可行解，计算 $Z$ 与峰值 $"peak"=max_h W_h$，按 $(Z, "peak")$ 字典序更新最优；
+ 30 日相互独立，使用 8 进程并行。

求解流程见@fig-flow-q1。

#figure(
  image("../../figures/fig_flow_q1.pdf", width: 78%),
  caption: [问题一求解流程],
) <fig-flow-q1>

== 结果与分析

求解耗时约 966 秒（8 进程并行），全部 30 日均达到最优。#strong[全月总工人数 $Z^((1)) = 10\,791$ 人]，等于理论下界 $sum_d L_d = 10\,791$，gap 为 0%；累计可行性 min slack = 0，binding 1 次；当日清零成立。

每日工人数与下界的对比见@fig-q1-workers。可以看到全部 30 日均贴合下界，下界紧。

#figure(
  image("../../figures/F-Q1-1_daily_workers.pdf", width: 88%),
  caption: [问题一每日工人数 $Z^((1))_d$ 与下界 $L_d$],
) <fig-q1-workers>

@fig-q1-typical 给出三个典型日（min/median/max）的小时进货 $v_h$ 与有效产能 $25 W_h$ 曲线，可见产能曲线总在累计意义下严格覆盖进货。

#figure(
  image("../../figures/F-Q1-2_typical_hourly.pdf", width: 90%),
  caption: [典型日小时进货与有效产能（min/median/max 三日）],
) <fig-q1-typical>

@fig-q1-start 给出 24 个候选起点在 30 日中被选用的频次。可见高峰前置时段（起点 0--3）出现频次最高，与夜间进货高峰一致。

#figure(
  image("../../figures/F-Q1-3_start_frequency.pdf", width: 88%),
  caption: [问题一中各班次起点被选用的天数],
) <fig-q1-start>

#strong[典型解读：] 以日总量最大的第 12 日为例，最优解为 5 个起点 $[0, 1, 2, 3, 16]$ 共 537 人，恰等于该日下界 $ceil.l 107\,377 slash 200 ceil.r = 537$；以总量最小的第 26 日为例，最优解为 5 起点 $[5, 7, 11, 20, 22]$ 共 296 人，亦达下界。
