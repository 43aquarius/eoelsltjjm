#let body-font = ("Times New Roman", "SimSun", "NSimSun", "Songti SC", "STSong")
#let song-font = ("SimSun", "NSimSun", "Songti SC", "STSong", "Times New Roman")
#let hei-font = ("Heiti SC", "STHeiti", "Songti SC", "STSong")
#let kai-font = ("KaiTi", "Kaiti SC", "STKaiti", "SimSun", "Songti SC")

#let cn-numbering(..nums) = {
  let ns = nums.pos()
  if ns.len() == 1 {
    numbering("一、", ns.at(0))
  } else if ns.len() == 2 {
    numbering("1.1", ns.at(0), ns.at(1))
  } else {
    numbering("1.1.1", ns.at(0), ns.at(1), ns.at(2))
  }
}

#set document(title: "物流分拣中心排班的整数规划与人员复用模型", author: ())
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  numbering: "1",
)
// 正文：宋体小四（12pt）
#set text(font: body-font, size: 12pt, lang: "zh")
#set par(
  first-line-indent: (amount: 2em, all: true),
  justify: true,
  leading: 1.0em,
  spacing: 0.95em,
)
#set heading(numbering: cn-numbering)
#set enum(numbering: "1.")
#set table(inset: 0.45em)
#show heading.where(level: 1): set align(center)
#show heading.where(level: 1): set text(size: 17.3pt, weight: "bold")
#show heading.where(level: 1): set block(above: 1.25em, below: 0.82em)
#show heading.where(level: 2): set text(size: 14.45pt, weight: "bold")
#show heading.where(level: 2): set block(above: 1.15em, below: 0.55em)
#show heading.where(level: 3): set text(size: 12.05pt, weight: "bold")
#show heading.where(level: 3): set block(above: 1.15em, below: 0.55em)
#show figure.caption: it => text(size: 12pt, weight: "bold")[#it]
#show raw: set text(size: 10pt, font: ("Courier New", "Menlo", "SimSun", "Songti SC"))
#show raw.where(block: true): set block(
  fill: luma(97%),
  stroke: 0.8pt + luma(70%),
  inset: 0.7em,
  above: 0.7em,
  below: 0.7em,
)

#let song = (body) => text(font: song-font, body)
#let hei = (body) => text(font: hei-font, weight: "bold", body)
#let kai = (body) => text(font: kai-font, body)
#let paper-title(body) = {
  align(center)[#text(size: 17.3pt, weight: "bold")[#body]]
  v(1em)
}
#let abstract-title() = align(center)[#text(size: 14pt, weight: "bold")[摘要]]
#let keywords-cn(body) = block(above: 1em)[
  #text(font: hei-font, size: 12pt, weight: "bold")[关键字：] #body
]
// 辽宁省赛摘要页表头：1 行 3 列无边框表格 + 下方一条横线
// 左：选择的题号 / B（20pt 红色）
// 中：2026 / 辽宁省大学生数学建模竞赛 / 摘要页（三行均小四）
// 右：校内编号 / 20（20pt 红色）
#let abstract-header(problem: "B", school-id: "20") = {
  block(width: 100%, above: 0pt, below: 0.4em)[
    #table(
      columns: (1fr, 2fr, 1fr),
      align: (center + horizon, center + horizon, center + horizon),
      stroke: none,
      inset: (x: 0.4em, y: 0.35em),
      [
        #text(size: 12pt)[选择的题号]\
        #text(size: 20pt, weight: "bold", fill: rgb("#c00000"))[#problem]
      ],
      [
        #text(size: 12pt)[2026]\
        #text(size: 12pt)[辽宁省大学生数学建模竞赛]\
        #text(size: 12pt)[摘要页]
      ],
      [
        #text(size: 12pt)[校内编号]\
        #text(size: 20pt, weight: "bold", fill: rgb("#c00000"))[#school-id]
      ],
    )
  ]
  // 分隔横线
  line(length: 100%, stroke: 0.8pt)
  v(0.8em)
}
#let abstract-cn(title, body, keywords, problem: "B", school-id: "20") = {
  abstract-header(problem: problem, school-id: school-id)
  align(center)[#text(size: 16pt, weight: "bold")[#title]]
  v(0.6em)
  abstract-title()
  block(above: 0.5em)[#body]
  keywords-cn(keywords)
  pagebreak()
}
#let toc-page() = {
  show outline.entry.where(level: 1): it => link(
    it.element.location(),
    block(above: 7pt)[
      #text(font: hei-font, size: 12pt, weight: "bold")[
        #grid(
          columns: (auto, 1fr, auto),
          column-gutter: 0.5em,
          [#it.prefix()#it.body()],
          [#repeat[.]],
          [#it.page()],
        )
      ]
    ],
  )
  outline(
    title: align(center)[#text(font: hei-font, size: 17.3pt, weight: "bold")[目录]],
    depth: 3,
  )
  pagebreak()
}
#let references-cn() = [
#heading(numbering: none, outlined: true)[参考文献]
#{ set par(first-line-indent: 0pt, spacing: 0.35em); include("references.typ") }
]
#let appendix-cn(file: "sections/A_code.typ") = [
#heading(numbering: none, outlined: true)[附录 A #h(1em) 核心代码]
#include(file)
]

#let three-line-table(caption, columns, header, body, inset: (x: 0.35em, y: 0.52em), cell-align: center) = {
  let col-count = header.len()
  let body-rows = calc.floor(body.len() / col-count)
  let bottom-y = body-rows + 1
  let styled-header = header.map(cell => strong(cell))

  block(width: 100%, breakable: false)[
    #align(center)[
      #box[
        #align(center)[#text(font: hei-font, size: 10.5pt, weight: "bold")[#caption]]
        #v(0.6em)
        #table(
          columns: columns,
          align: cell-align,
          stroke: none,
          inset: inset,
          table.hline(y: 0, stroke: 0.8pt),
          table.hline(y: 1, stroke: 0.5pt),
          table.hline(y: bottom-y, stroke: 0.8pt),
          ..styled-header,
          ..body,
        )
      ]
    ]
  ]
}

#counter(page).update(1)

#abstract-cn(
  [物流分拣中心排班的整数规划与人员复用模型],
  [
本文针对物流分拣中心 30 日逐小时进货量数据，建立基于整数规划与列式生成的排班模型，依次回答三个子问题。

针对#strong[问题一]，将"每天 5 个 8 小时连续班次、每小时 25 件处理能力"建模为以 24 个候选起点为标号的整数规划：决策变量为各起点选用指示 $delta_s$ 和当日工人数 $xi_s$，目标为日工人总数最少，约束为每小时累计处理量不少于累计进货量（保证当日清零）。求解时枚举 $C(24,5)=42504$ 个起点组合并对各组合解 5 变量线性规划，并以"峰值时段在岗人数最低"为二级目标。30 日均达到理论下界 $L_d=ceil.l "total"_d/200 ceil.r$，#strong[全月总工人数为 10\,791 人]。

针对#strong[问题二]，在问题一基础上引入 0--12 时段进货须在 16 点前累计处理完的截止约束，以及"每名工人在所在班次的 8 小时中有 1 小时只能处理 10 件"的低产能约束。引入非负连续变量 $y_(s,h)$ 表示班次 $s$ 中低产能小时为 $h$ 的工人数，与 $sum_h y_(s,h)=xi_s$ 关联，每小时有效产能写为 $25 sum_s A_(s,h) xi_s - 15 sum_s y_(s,h)$；用 PuLP+CBC 逐日求解。#strong[全月总工人数为 11\,668 人]，较问题一上升 8.13%，与有效产能从 200 件/班降至 185 件/班的解析估计吻合；累计可行性与 16 点截止约束全部满足。

针对#strong[问题三]，将问题二输出的每日总人数 $D_d$ 视为月度需求，工人采用"出勤模式"（30 位长度、含 23 个 1 且最长连续 1 不超过 7）的列式生成方法：先穷举 767\,544 条合法模式，对所有模式解 LP 松弛得 LP 最优值 581，再在 LP 活跃的 14 条模式上做 MILP 精化得到整数解。#strong[全月最少招工人数为 N\=581]，恰等于下界 $max(ceil.l sum D / 23 ceil.r, max_d D_d)$；每人 30 日恰工作 23 天、任意 8 日窗口工作日不超过 7 天，全部满足。最长连续工作天数为 7 天，共 282 人；6 天 243 人；5 天 56 人。

三问最优解均达到对应理论下界，模型对单人产能、连续工作上限等参数的灵敏度低；班次起点在问题二中全月自发收敛为统一的 $[0, 1, 9, 14, 23]$，体现了模型的稳健性与可解释性。
],
[排班优化 #h(1em) 整数规划 #h(1em) 列式生成 #h(1em) 班次设计 #h(1em) 物流分拣],
problem: "B",
school-id: "20",
)

#toc-page()

#include("sections/1_restatement.typ")
#include("sections/2_analysis.typ")
#include("sections/3_assumptions.typ")
#include("sections/4_symbols.typ")
#include("sections/5_problem1.typ")
#include("sections/6_problem2.typ")
#include("sections/7_problem3.typ")
#include("sections/8_sensitivity.typ")
#include("sections/9_evaluation.typ")

#pagebreak()
#references-cn()
#pagebreak()
#appendix-cn()
