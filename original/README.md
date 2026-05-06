# 上市公司制造业企业AI应用评估项目

基于BERT的多任务学习模型，用于识别和分类企业年报中的AI应用信息。该项目使用中文BERT预训练模型，同时预测三个相关任务：是否存在AI应用、AI使用方式和AI应用类型。

## 项目概述

本项目实现了一个多任务学习框架，通过共享BERT编码器来同时学习三个分类任务：

1. **是否存在AI应用**：判断文本中是否提及AI技术的应用
2. **AI使用方式**：识别AI技术的具体使用方式（如自身应用、产品输出等）
3. **AI应用类型**：分类AI应用的具体类型（如生产型、决策型等）

通过多任务学习，模型能够利用任务间的相关性，提高整体预测性能。该项目可用于分析上市公司制造业企业的数字化转型程度和AI应用情况。

## 技术栈

- **深度学习框架**：PyTorch
- **预训练模型**：BERT (bert-base-chinese)
- **模型库**：Hugging Face Transformers
- **数据处理**：Pandas, scikit-learn
- **其他工具**：tqdm (进度条), openpyxl (Excel读取)

## 项目结构

```
.
├── data.py                      # 数据加载和预处理模块
├── model.py                     # BERT多任务模型定义
├── train.py                     # 模型训练脚本
├── evaluate_models.py           # 模型评估脚本
├── predict.py                   # 模型预测脚本
├── requirements.txt             # 项目依赖
├── 合并结果.xlsx                # 训练数据文件
└── models/                      # 保存训练好的模型
    ├── bert_multitask_epoch1.pt
    ├── bert_multitask_epoch2.pt
    └── ...
```

## 数据说明

项目使用 `合并结果.xlsx` 文件作为训练数据，包含以下字段：

- **股票代码**：上市公司股票代码
- **年份**：数据所属年份
- **句子**：从年报中提取的句子（模型输入）
- **所属年报总句子数**：该年报的总句子数
- **是否存在AI应用**：标记句子是否提及AI应用（任务1标签）
- **AI使用方式**：AI的使用方式（任务2标签）
- **AI应用类型**：AI的应用类型（任务3标签）

## 核心模块说明

### 1. data.py - 数据处理模块

提供数据加载、预处理和数据集创建功能：

- `load_data()`: 从Excel文件加载数据
- `preprocess_data()`: 数据清洗，处理缺失值
- `encode_labels()`: 使用LabelEncoder对三个任务的标签进行编码
- `create_data_loaders()`: 创建训练集和测试集的DataLoader（默认8:2分割）
- `AIDataset`: 自定义PyTorch数据集类，处理文本tokenization

### 2. model.py - 模型定义

`BertMultiTask`类实现了多任务学习架构：

- 共享的BERT编码器提取文本特征
- 三个独立的分类头分别处理三个任务
- 使用[CLS] token的表示作为句子级特征

### 3. train.py - 模型训练

训练流程：

- 加载并预处理数据
- 初始化BERT多任务模型
- 使用AdamW优化器和交叉熵损失函数
- 训练5个epoch，每个epoch后保存模型
- 总损失为三个任务损失之和

**训练参数**：
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 5
- 优化器: AdamW
- 最大序列长度: 128

### 4. evaluate_models.py - 模型评估

评估功能：

- 加载训练好的模型
- 在测试集上评估性能
- 计算每个任务的准确率、精确率、召回率和F1分数
- 提供每个标签的详细指标
- 支持批量评估多个模型

### 5. predict.py - 模型预测

预测功能：

- 加载训练好的模型
- 对新文本进行预测
- 返回三个任务的预测结果
- 支持批量预测

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- transformers
- torch
- pandas
- scikit-learn
- tqdm
- openpyxl

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程会：
- 自动加载`合并结果.xlsx`数据文件
- 下载bert-base-chinese预训练模型（使用HuggingFace镜像站）
- 训练5个epoch
- 每个epoch后保存模型到`models/`目录

### 2. 评估模型

```bash
python evaluate_models.py
```

评估脚本会：
- 加载指定的模型文件（epoch1和epoch2）
- 在测试集上评估性能
- 输出详细的评估指标

### 3. 预测新文本

```bash
python predict.py
```

或在代码中使用：

```python
from predict import predict

texts = [
    '公司引入智能生产线，实现生产自动化',
    '我们使用大数据分析技术优化决策流程'
]

model_path = 'models/bert_multitask_epoch5.pt'
results = predict(texts, model_path)

for result in results:
    print(f"文本: {result['text']}")
    print(f"是否存在AI应用: {result['是否存在AI应用']}")
    print(f"AI使用方式: {result['AI使用方式']}")
    print(f"AI应用类型: {result['AI应用类型']}")
```

## 模型架构

```
输入文本
    ↓
BERT Tokenizer
    ↓
BERT Encoder (共享)
    ↓
[CLS] Token表示
    ↓
    ├─→ 分类头1 → 是否存在AI应用
    ├─→ 分类头2 → AI使用方式
    └─→ 分类头3 → AI应用类型
```

多任务学习的优势：
- 共享底层特征表示，提高数据利用效率
- 任务间相互促进，提升整体性能
- 减少过拟合风险

## 性能优化

项目包含以下优化配置：

- 使用HuggingFace镜像站加速模型下载
- 支持GPU加速训练和推理
- 批量处理提高效率
- 进度条显示训练进度
- 自动设备检测（CUDA/CPU）

## 评估指标

模型评估包括以下指标：

- **准确率 (Accuracy)**: 整体预测正确的比例
- **精确率 (Precision)**: 预测为正类中实际为正类的比例
- **召回率 (Recall)**: 实际为正类中被正确预测的比例
- **F1分数**: 精确率和召回率的调和平均
- **各标签详细指标**: 每个类别的独立评估

## 注意事项

1. **SSL验证**：代码中禁用了SSL验证（仅用于开发环境），生产环境建议启用
2. **数据路径**：确保`合并结果.xlsx`文件在项目根目录
3. **GPU支持**：如果有CUDA可用，模型会自动使用GPU加速
4. **内存占用**：BERT模型较大，建议至少8GB内存
5. **首次运行**：首次运行会下载预训练模型，可能需要较长时间

## 扩展建议

1. **超参数调优**：调整学习率、batch size、epoch数等
2. **模型改进**：尝试其他预训练模型（RoBERTa、ELECTRA等）
3. **数据增强**：使用回译、同义词替换等技术扩充训练数据
4. **损失函数优化**：为不同任务设置不同权重
5. **集成学习**：结合多个模型的预测结果
6. **特征工程**：添加企业特征、行业特征等辅助信息

## 应用场景

- 上市公司AI应用情况统计分析
- 制造业数字化转型程度评估
- 行业AI技术应用趋势研究
- 企业年报自动化分析
- 投资决策辅助分析

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎提出Issue。
