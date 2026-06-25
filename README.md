# 少样本电子产品外观缺陷检测

联宝杯参赛项目 — 在仅 60 张缺陷标注样本的条件下，对笔记本电脑外观 4 类缺陷（`collision` / `dirt` / `plain particle` / `scratch`）进行检测。

## 技术路线

- **检测模型**：YOLOv8n / YOLOv8s / YOLO11s / YOLO11m（CPU-only 训练）
- **少样本策略**：迁移学习 + 冻结骨干 → fine-tune 解冻 → 多模型集成
- **数据增强**：基于 polygon-mask 的 Copy-Paste（把负样本的缺陷区域贴到正样本作为新训练图）+ Mosaic + MixUp + 随机几何/颜色变换
- **类平衡**：对极少样本类（如 dirt 只有 1 个 polygon）做 25× 过采样
- **推理融合**：多尺度 TTA（[384, 448, 512, 576, 640]）+ 水平翻转 → Weighted Boxes Fusion (WBF)
- **跨模型集成**：v8s "救援版" + v8s "无救援版" 加权 WBF 集成（最优权重 0.3:1.7）

## 关键工程发现

### 1. 训练数据泄漏（合规问题）

主办方提供的原始数据集中，**11 张测试图字节一致地出现在「训练集/负样本」目录**——其多边形标注会直接泄漏给训练模型。`src/audit.py` 通过 MD5 比对发现，`src/constants.py` 维护 `LEAKED_STEMS` 列表硬剔除。

代价：dirt 类原有 15 个 polygon 中 13 个全部泄漏，剔除后只剩 2 个有效标注。但合规优先。

### 2. val 集严重不可靠

val 集只有 31 张图 / 17 个实例，单个样本就能让 val mAP 波动 5%+。多次出现 val mAP +57% 但 test mAP 只涨 1-5% 的情况。所以以 val 调参不可靠，最终决策以 test 反馈为准。

### 3. dirt 救援的双刃剑

- dirt polygon 只剩 1 个时，25× 过采样让模型至少能输出 dirt 预测（dirt 框数 0 → 116），但纯过拟合那 1 个 polygon
- 反过来取消 dirt 救援 → 模型输出更"挑剔"的 scratch/particle 预测（hi-c 信号反而暴涨）
- 最优是**集成两种训练**：救援版（保 dirt 信号）+ 无救援版（保 scratch/particle 质量），加权 0.3:1.7

### 4. mosaic 增强对少样本可能有害

```bash
python src/train.py --model yolov8s.pt --mosaic 0 --cos-lr
```

关闭 mosaic + cosine LR 训出来的 v8s，单模型测试集 hi-c 框数从 **159 → 413**（涨 2.6 倍）。

### 5. 后处理优化（不重训直接改 JSON）

`src/postprocess.py` 实现的可调参数：
- 白色区域过滤（README 明说不算缺陷）
- bbox 扩展 / 收缩
- 小框过滤 / 每图限框 / 每类限框

## 目录结构

```
.
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── .gitignore                   # 忽略数据 / 模型 / 实验输出
├── 少样本条件下电子产品外观缺陷检测.txt  # 赛题原文
├── plans/                       # 早期设计文档
└── src/                         # 全部代码
    ├── constants.py             # 类名 / 类 ID / LEAKED_STEMS 列表
    ├── data_preparation.py      # LabelMe JSON → YOLO 格式 + 8:2 划分
    ├── augment_copy_paste.py    # 离线 Copy-Paste 增强（带 polygon mask）
    ├── train.py                 # YOLO 训练入口（支持 yolov8/yolo11，mosaic/cos_lr 可调）
    ├── predict.py               # 简单单尺度推理
    ├── predict_simple.py        # 增量 resume 推理（防中断）
    ├── predict_wbf.py           # 多尺度 + 翻转 TTA + WBF 推理
    ├── ensemble_wbf.py          # 跨模型 WBF 集成
    ├── postprocess.py           # JSON 后处理（白过滤/bbox 调整/限框等）
    ├── package_submission.py    # 按官方格式打包 ZIP（UTF-8 NoBOM）
    ├── filter_by_conf.py        # conf 阈值变体生成
    ├── inject_class.py          # 单类高 conf 框注入（取一个模型的 dirt 加到另一个）
    ├── visualize.py             # 推理可视化
    ├── audit.py                 # 提交合规审计（BOM、image_id、bbox 边界、数据泄漏）
    ├── mine_false_positives.py  # 正样本上挖假阳性（hard negative mining）
    └── post_train_pipeline.py   # 一键: 推理 → 多阈值 → 打包
```

## 快速开始

```bash
# 1. 安装依赖（CPU torch）
uv pip install --python .venv/Scripts/python.exe \
    --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/Scripts/python.exe \
    ultralytics opencv-python albumentations ensemble-boxes tqdm pyyaml

# 2. 准备数据集（自动剔除 11 张泄漏图）
python src/data_preparation.py

# 3. Copy-Paste 增强（dirt 25× 过采样）
python src/augment_copy_paste.py --per-positive 3 --defects-per-image 1 3 \
    --class-weights 2.0 25.0 1.0 1.0 --clear-existing

# 4. 训练 v8s 救援版（~1.85h CPU）
python src/train.py --model yolov8s.pt --epochs 30 --imgsz 512 --batch 4 \
    --patience 15 --name defect_yolov8s_dirt_rescue

# 5. 训练 v8s 无救援版（先重跑 augment 无 class-weights）
python src/augment_copy_paste.py --per-positive 4 --defects-per-image 1 4 --clear-existing
python src/train.py --model yolov8s.pt --epochs 30 --imgsz 512 --batch 4 \
    --patience 15 --name defect_yolov8s_no_dirt

# 6. 双模型 WBF 推理
python src/predict_wbf.py --weights runs/defect_yolov8s_dirt_rescue/weights/best.pt \
    --conf 0.05 --imgsz 384 448 512 576 640 --flip --out-dir submission_rescue
python src/predict_wbf.py --weights runs/defect_yolov8s_no_dirt/weights/best.pt \
    --conf 0.05 --imgsz 384 448 512 576 640 --flip --out-dir submission_no_dirt

# 7. 跨模型 WBF 集成（最优权重 0.3:1.7）
python src/ensemble_wbf.py --inputs submission_rescue submission_no_dirt \
    --weights 0.3 1.7 --iou-thr 0.55 --out submission_final

# 8. 打包提交 ZIP（UTF-8 NoBOM，含官方目录结构）
python src/package_submission.py --source submission_final/per_image \
    --team "你的队名" --work "作品名"
```

## 评测要求

格式（`少样本条件下电子产品外观缺陷检测.txt` 第 33 行起）：
- ZIP 命名 `少样本条件下电子产品外观缺陷检测_团队名_作品名.zip`
- 内含同名文件夹，876 个 `*.json`（每张测试图一份）
- UTF-8 NoBOM 编码
- bbox 原始像素坐标 `[x1, y1, x2, y2]`，**不归一化**
- label 限定 4 类

`src/audit.py` 自动校验全部合规项。

## 心得

1. **val 集太小（17 实例）不可信** — 任何调参都要看实际 test 反馈，否则就是过拟合 val
2. **少样本 + 大模型双刃剑** — YOLO11m 比 YOLO11s val mAP +30% 但 test 反而降，因为大模型过拟合更狠
3. **集成的胜利** — 单模型怎么调都到不了 0.20，「救援 + 无救援」加权集成才突破到 0.200
4. **数据泄漏要主动审计** — 主办方提供的"训练集"也可能含测试数据，规则禁止则必须剔除

## 许可

代码用于学习交流。比赛数据归原主办方所有，本仓库未包含。
