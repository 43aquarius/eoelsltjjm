= 问题二的模型建立与求解

== 新增约束的建模

问题二在问题一基础上引入两项新约束。

#strong[（a）16 点截止约束。] 按 H4 的累计口径，0--11 时段累计进货须在 15 时末累计完成处理，即
$ sum_(h=0)^(15) "Cap"_(d,h) gt.eq sum_(h=0)^(11) v_(d,h), quad forall d. $
该约束被"每个时刻累计可行性"（D2）所蕴含，因而在代码中只显式声明 D2，并在校验阶段单独输出该项 slack。

#strong[（b）班内 1 小时低产能。] 每个班次 $s$ 在被使用时含 $xi_(d,s)$ 名工人，每人在该 8 小时班次中有 1 小时只能处理 10 件。引入非负变量 $y_(d,s,h)$ 表示第 $d$ 日班次 $s$ 中"低产能小时为 $h$"的工人数（$h in "Shift"(s)$）。两条耦合关系：
$ sum_(h in "Shift"(s)) y_(d,s,h) = xi_(d,s), quad forall d, s, $
$ "Cap"_(d,h) = c dot W_(d,h) - (c - c_"low") dot sum_(s: h in "Shift"(s)) y_(d,s,h). $

注意 $y$ 取连续值不会改变最优 $Z^((2))$ 的整数性，因为 $xi$ 为整数且 $sum_h y_(s,h) = xi_s$。在实施中将 $y$ 放宽为非负连续变量可显著加速 CBC 求解，并不损失最优值。

== 模型完整形式

记完整 MILP（每日独立求解）：

$ &min space sum_(s=0)^(23) xi_(d,s) \
&"s.t." sum_(s=0)^(23) delta_s = 5, quad xi_(d,s) lt.eq M delta_s, \
&sum_(h in "Shift"(s)) y_(d,s,h) = xi_(d,s), quad forall s, \
&sum_(h=0)^(H) "Cap"_(d,h) gt.eq sum_(h=0)^(H) v_(d,h), quad forall H in {0,...,23}, \
&delta_s in {0,1}, quad xi_(d,s) in bb(Z)_+, quad y_(d,s,h) gt.eq 0. $

求解流程见@fig-flow-q2。

#figure(
  image("../../figures/fig_flow_q2.pdf", width: 80%),
  caption: [问题二求解流程],
) <fig-flow-q2>

== 求解结果

使用 PuLP + CBC 逐日求解（threads=2, timeLimit=30s），总耗时约 15 秒。求解状态全部为 Optimal。#strong[全月总工人数 $Z^((2)) = 11\,668$ 人]，较 $Z^((1)) = 10\,791$ 上升 877 人（+8.13%），与"有效产能从 200 件/班 ($c dot 8$) 降至 185 件/班 ($25 dot 7 + 10$) 即 $-7.5%$"的解析估计吻合。30 日全部满足 $Z^((2))_d gt.eq Z^((1))_d$，模型单调性成立。

@fig-q2-cum 给出三个典型日（按 $Z^((2))$ 排序的 min/median/max）的累计进货曲线、累计有效产能曲线，以及 16:00 截止垂线。可见两条曲线全程贴合且 16:00 时刻有较大的安全裕度。

#figure(
  image("../../figures/F-Q2-1_cumulative_deadline.pdf", width: 90%),
  caption: [问题二典型日累计进货与累计有效产能（标注 16:00 截止）],
) <fig-q2-cum>

@fig-q2-vs 对比 30 日 $Z^((1))_d$ 与 $Z^((2))_d$，可见 $Z^((2))$ 全程位于 $Z^((1))$ 上方，平均 +29 人/日。

#figure(
  image("../../figures/F-Q2-2_Q2_vs_Q1.pdf", width: 88%),
  caption: [问题二与问题一的每日工人数对比],
) <fig-q2-vs>

@fig-q2-low 给出 30 日累计的低产能工人在 24 小时上的分布。可见排班器倾向将低产能时段集中在进货量谷段（6 点、13 点），与直觉一致。

#figure(
  image("../../figures/F-Q2-3_low_cap_hour_distribution.pdf", width: 88%),
  caption: [30 日累计的低产能工人在 24 小时上的分布],
) <fig-q2-low>

#strong[起点的一致性：] 30 日中 29 日的最优起点为 $[0, 1, 9, 14, 23]$，仅第 11 日为 $[1, 4, 9, 14, 23]$（轻微偏移）。这表明在问题二的新约束下，最优解自发收敛到一组全月统一的班次起点，强化了"统一作息"的可解释性，也为问题三的人员复用提供了天然基础。

#strong[校验：] 累计可行性 min slack = 2；16 点截止约束的最小 slack 为 13\,030 件，远离 binding，说明在累计口径下该约束并非紧约束；当日清零、Q2 $gt.eq$ Q1 单调性等检验全部通过。
