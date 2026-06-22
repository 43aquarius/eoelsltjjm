#v(0.5em)

== 支撑材料文件列表

随论文一同提交的支撑材料按目录结构如下：

#text(size: 11pt)[
```
支撑材料/
├── code/                            # 全部可运行源代码（Python 3.14）
│   ├── utils.py                     # 数据加载 / 覆盖矩阵 / 下界 / 字体配置
│   ├── eda.py                       # 数据探查与摘要 JSON
│   ├── problem1.py                  # 问题一 (枚举 C(24,5)+LP，多进程)
│   ├── problem2.py                  # 问题二 (MILP，CBC，连续 y)
│   ├── problem3.py                  # 问题三 (模式生成 + LP + MILP)
│   ├── fig_q1.py / fig_q2.py / fig_q3.py     # 各问题数据图
│   ├── fig_drawio_pdf.py            # 非数据图共享样式
│   ├── fig_roadmap.py               # 技术路线图
│   └── fig_flow_q1_pdf.py / fig_flow_q2_pdf.py / fig_flow_q3_pdf.py / fig_model_pdf.py
│
├── results/                         # 求解结果 JSON
│   ├── eda_summary.json             # 数据摘要
│   ├── q1_solution.json             # 问题一最优解 / 校验 / 下界
│   ├── q2_solution.json             # 问题二最优解 / y 矩阵 / D3 校验
│   └── q3_solution.json             # 问题三 N、出勤矩阵、模式
│
└── figures/                         # 全部 PDF 图与可编辑源
    ├── F-EDA-{1,2,3}*.pdf           # 数据探查
    ├── F-Q{1,2,3}-{1,2,3}*.pdf      # 各问题结果图
    ├── fig_roadmap.{drawio,pdf}     # 技术路线
    ├── fig_flow_q{1,2,3}.{drawio,pdf}  # 三问求解流程
    └── fig_model.{drawio,pdf}       # 模型变量关系
```
]

复现命令：

#text(size: 11pt)[
```bash
PY="C:/Users/TC/AppData/Local/Python/pythoncore-3.14-64/python.exe"
"$PY" -m pip install openpyxl pandas numpy scipy pulp matplotlib pypdfium2
"$PY" code/utils.py     && "$PY" code/eda.py
"$PY" code/problem1.py  && "$PY" code/problem2.py  && "$PY" code/problem3.py
"$PY" code/fig_q1.py    && "$PY" code/fig_q2.py    && "$PY" code/fig_q3.py
"$PY" code/fig_roadmap.py
"$PY" code/fig_flow_q1_pdf.py && "$PY" code/fig_flow_q2_pdf.py && "$PY" code/fig_flow_q3_pdf.py
"$PY" code/fig_model_pdf.py
typst compile --root . paper/main.typ
```
]

== 完整源代码

下列代码按调用依赖排序，全部可直接运行复现本文所有结果。

=== `code/utils.py`

#raw(read("../../code/utils.py"), lang: "python", block: true)

=== `code/eda.py`

#raw(read("../../code/eda.py"), lang: "python", block: true)

=== `code/problem1.py`

#raw(read("../../code/problem1.py"), lang: "python", block: true)

=== `code/problem2.py`

#raw(read("../../code/problem2.py"), lang: "python", block: true)

=== `code/problem3.py`

#raw(read("../../code/problem3.py"), lang: "python", block: true)

=== `code/fig_q1.py`

#raw(read("../../code/fig_q1.py"), lang: "python", block: true)

=== `code/fig_q2.py`

#raw(read("../../code/fig_q2.py"), lang: "python", block: true)

=== `code/fig_q3.py`

#raw(read("../../code/fig_q3.py"), lang: "python", block: true)

=== `code/fig_drawio_pdf.py`

#raw(read("../../code/fig_drawio_pdf.py"), lang: "python", block: true)

=== `code/fig_roadmap.py`

#raw(read("../../code/fig_roadmap.py"), lang: "python", block: true)

=== `code/fig_flow_q1_pdf.py`

#raw(read("../../code/fig_flow_q1_pdf.py"), lang: "python", block: true)

=== `code/fig_flow_q2_pdf.py`

#raw(read("../../code/fig_flow_q2_pdf.py"), lang: "python", block: true)

=== `code/fig_flow_q3_pdf.py`

#raw(read("../../code/fig_flow_q3_pdf.py"), lang: "python", block: true)

=== `code/fig_model_pdf.py`

#raw(read("../../code/fig_model_pdf.py"), lang: "python", block: true)
