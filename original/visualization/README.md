# 模型可视化对比

本目录包含四个BERT多任务模型（Epoch 1-4）的性能对比可视化代码。

## 文件说明

- `plot_model_comparison.py`: 主要的可视化脚本，用于评估模型并生成对比图表
- `plots/`: 生成的图表保存目录

## 使用方法

1. 确保已安装所需依赖：
```bash
pip install matplotlib numpy torch transformers scikit-learn tqdm
```

2. 运行可视化脚本：
```bash
cd original/visualization
python plot_model_comparison.py
```

## 生成的图表

脚本会在 `plots/` 目录下生成以下图表：

### 1. 各任务指标对比图（3张）
- `task1_metrics_comparison.png`: 任务1（是否存在AI应用）的四个指标对比
- `task2_metrics_comparison.png`: 任务2（AI使用方式）的四个指标对比
- `task3_metrics_comparison.png`: 任务3（AI应用类型）的四个指标对比

每张图展示四个模型在准确率、精确率、召回率、F1分数上的表现。

### 2. 准确率总览图（1张）
- `all_tasks_accuracy_comparison.png`: 四个模型在三个任务上的准确率对比

### 3. F1分数总览图（1张）
- `all_tasks_f1_comparison.png`: 四个模型在三个任务上的F1分数对比

### 4. 雷达图（4张）
- `radar_model_1.png`: Epoch 1模型的三任务性能雷达图
- `radar_model_2.png`: Epoch 2模型的三任务性能雷达图
- `radar_model_3.png`: Epoch 3模型的三任务性能雷达图
- `radar_model_4.png`: Epoch 4模型的三任务性能雷达图

每张雷达图展示单个模型在三个任务上的四个指标表现。

### 5. 训练进度趋势图（1张）
- `training_progress.png`: 展示四个指标随训练轮次（Epoch 1-4）的变化趋势

### 6. 评估结果JSON
- `evaluation_results.json`: 所有模型的详细评估指标数据

## 注意事项

- 脚本会使用前50000条数据进行评估
- 需要GPU支持以加快评估速度（可选）
- 确保 `../合并结果.xlsx` 数据文件存在
- 确保 `../models/` 目录下有四个模型文件
