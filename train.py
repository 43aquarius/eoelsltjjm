import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.optim import AdamW
from tqdm import tqdm

from data import load_data, preprocess_data, encode_labels, create_data_loaders
from model import BertMultiTask

def train_model():
    # 配置参数
    batch_size = 16
    epochs = 5
    learning_rate = 2e-5
    model_name = 'bert-base-chinese'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载和预处理数据
    print('加载数据...')
    df = load_data('合并结果.xlsx')
    df = preprocess_data(df)
    df, le1, le2, le3 = encode_labels(df)
    
    # 获取标签数量
    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)
    
    print(f'标签数量: 存在AI应用={num_labels1}, AI使用方式={num_labels2}, AI应用类型={num_labels3}')
    print(f'存在AI应用标签: {le1.classes_}')
    print(f'AI使用方式标签: {le2.classes_}')
    print(f'AI应用类型标签: {le3.classes_}')
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(df, tokenizer, batch_size=batch_size)
    print(f'训练集大小: {len(train_loader.dataset)}, 测试集大小: {len(test_loader.dataset)}')
    
    # 初始化模型
    model = BertMultiTask(num_labels1, num_labels2, num_labels3, model_name)
    model.to(device)
    
    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print('开始训练...')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            label3 = batch['label3'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            logits1, logits2, logits3 = model(input_ids, attention_mask)
            
            # 计算损失
            loss1 = criterion(logits1, label1)
            loss2 = criterion(logits2, label2)
            loss3 = criterion(logits3, label3)
            
            # 总损失
            loss = loss1 + loss2 + loss3
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, 平均损失: {avg_loss:.4f}')
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/bert_multitask_epoch{epoch+1}.pt')
        print(f'模型已保存到 models/bert_multitask_epoch{epoch+1}.pt')
    
    print('训练完成！')

if __name__ == '__main__':
    train_model()
