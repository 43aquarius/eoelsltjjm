import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import load_data, preprocess_data, encode_labels, create_data_loaders
from model import BertMultiTask

def evaluate_model(model_path):
    # 配置参数
    batch_size = 16
    model_name = 'bert-base-chinese'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载和预处理数据
    print('加载数据...')
    df = load_data('合并结果.xlsx')
    df = preprocess_data(df)
    df, le1, le2, le3 = encode_labels(df)
    
    # 只使用前5000条数据进行测试，加快评估速度
    df = df.head(5000)
    print(f'使用数据量: {len(df)}')
    
    # 获取标签数量
    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 创建数据加载器
    _, test_loader = create_data_loaders(df, tokenizer, batch_size=batch_size)
    print(f'测试集大小: {len(test_loader.dataset)}')
    
    # 初始化模型
    model = BertMultiTask(num_labels1, num_labels2, num_labels3, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 存储预测结果和真实标签
    all_labels1 = []
    all_preds1 = []
    all_labels2 = []
    all_preds2 = []
    all_labels3 = []
    all_preds3 = []
    
    print('开始评估...')
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            label3 = batch['label3'].to(device)
            
            # 前向传播
            logits1, logits2, logits3 = model(input_ids, attention_mask)
            
            # 获取预测结果
            preds1 = torch.argmax(logits1, dim=1)
            preds2 = torch.argmax(logits2, dim=1)
            preds3 = torch.argmax(logits3, dim=1)
            
            # 存储结果
            all_labels1.extend(label1.cpu().numpy())
            all_preds1.extend(preds1.cpu().numpy())
            all_labels2.extend(label2.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_labels3.extend(label3.cpu().numpy())
            all_preds3.extend(preds3.cpu().numpy())
    
    # 计算评估指标
    def calculate_metrics(labels, preds, label_names):
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        print(f'准确率: {accuracy:.4f}')
        print(f'精确率: {precision:.4f}')
        print(f'召回率: {recall:.4f}')
        print(f'F1分数: {f1:.4f}')
        print('各标签详细指标:')
        # 使用labels参数指定所有可能的类别，确保为每个标签计算指标
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=range(len(label_names)))
        for i, label in enumerate(label_names):
            print(f'标签 {label}: 精确率={precision[i]:.4f}, 召回率={recall[i]:.4f}, F1={f1[i]:.4f}, 支持数={support[i]}')
    
    # 评估第一个任务：是否存在AI应用
    print('评估任务1: 是否存在AI应用')
    calculate_metrics(all_labels1, all_preds1, le1.classes_)
    
    # 评估第二个任务：AI使用方式
    print('评估任务2: AI使用方式')
    calculate_metrics(all_labels2, all_preds2, le2.classes_)
    
    # 评估第三个任务：AI应用类型
    print('评估任务3: AI应用类型')
    calculate_metrics(all_labels3, all_preds3, le3.classes_)

if __name__ == '__main__':
    # 评估两个模型
    model_paths = [
        'models/bert_multitask_epoch1.pt',
        'models/bert_multitask_epoch2.pt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f'评估模型: {model_path}')
            evaluate_model(model_path)
        else:
            print(f'模型文件不存在: {model_path}')