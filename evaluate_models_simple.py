import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

class AIDataset(Dataset):
    def __init__(self, texts, labels1, labels2, labels3, tokenizer, max_length=128):
        self.texts = texts
        self.labels1 = labels1
        self.labels2 = labels2
        self.labels3 = labels3
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]
        label3 = self.labels3[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label1': torch.tensor(label1, dtype=torch.long),
            'label2': torch.tensor(label2, dtype=torch.long),
            'label3': torch.tensor(label3, dtype=torch.long)
        }

class BertMultiTask(nn.Module):
    def __init__(self, num_labels1, num_labels2, num_labels3, model_name='bert-base-chinese'):
        super(BertMultiTask, self).__init__()
        # 加载预训练的Bert模型
        self.bert = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        
        # 获取Bert的隐藏层维度
        self.hidden_size = self.bert.config.hidden_size
        
        # 三个分类头
        self.classifier1 = nn.Linear(self.hidden_size, num_labels1)
        self.classifier2 = nn.Linear(self.hidden_size, num_labels2)
        self.classifier3 = nn.Linear(self.hidden_size, num_labels3)
    
    def forward(self, input_ids, attention_mask):
        # 获取Bert的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS] token的表示作为句子表示
        cls_output = outputs.hidden_states[-1][:, 0, :]
        
        # 三个分类任务的输出
        logits1 = self.classifier1(cls_output)
        logits2 = self.classifier2(cls_output)
        logits3 = self.classifier3(cls_output)
        
        return logits1, logits2, logits3

def evaluate_model(model_path, test_loader, device, num_labels1, num_labels2, num_labels3):
    """评估模型"""
    # 初始化模型
    model = BertMultiTask(num_labels1, num_labels2, num_labels3)
    model.to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 存储预测结果和真实标签
    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_labels1 = []
    all_labels2 = []
    all_labels3 = []
    
    # 评估过程
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='评估中'):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            label3 = batch['label3'].to(device)
            
            # 前向传播
            logits1, logits2, logits3 = model(input_ids, attention_mask)
            
            # 获取预测结果
            preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
            preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
            preds3 = torch.argmax(logits3, dim=1).cpu().numpy()
            
            # 存储结果
            all_preds1.extend(preds1)
            all_preds2.extend(preds2)
            all_preds3.extend(preds3)
            all_labels1.extend(label1.cpu().numpy())
            all_labels2.extend(label2.cpu().numpy())
            all_labels3.extend(label3.cpu().numpy())
    
    # 计算准确率
    accuracy1 = accuracy_score(all_labels1, all_preds1)
    accuracy2 = accuracy_score(all_labels2, all_preds2)
    accuracy3 = accuracy_score(all_labels3, all_preds3)
    
    print(f'\n模型: {os.path.basename(model_path)}')
    print(f'存在AI应用准确率: {accuracy1:.4f}')
    print(f'AI使用方式准确率: {accuracy2:.4f}')
    print(f'AI应用类型准确率: {accuracy3:.4f}')
    print(f'平均准确率: {(accuracy1 + accuracy2 + accuracy3) / 3:.4f}')
    
    return accuracy1, accuracy2, accuracy3

def main():
    # 配置参数
    batch_size = 32
    model_name = 'bert-base-chinese'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载和预处理数据
    print('加载数据...')
    df = pd.read_excel('合并结果.xlsx')
    df = df.fillna('不适用')
    df = df[df['句子'].notna() & (df['句子'] != '')]
    
    # 使用前5000条数据进行测试
    df = df.head(5000)
    print(f'使用数据量: {len(df)}')
    
    # 编码标签
    print('编码标签...')
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    
    df['label1'] = le1.fit_transform(df['是否存在AI应用'])
    df['label2'] = le2.fit_transform(df['AI使用方式'])
    df['label3'] = le3.fit_transform(df['AI应用类型'])
    
    # 获取标签数量
    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)
    
    print(f'标签数量: 存在AI应用={num_labels1}, AI使用方式={num_labels2}, AI应用类型={num_labels3}')
    
    # 初始化tokenizer
    print('初始化tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 创建测试数据集
    print('创建测试数据集...')
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    test_dataset = AIDataset(
        test_df['句子'].tolist(),
        test_df['label1'].tolist(),
        test_df['label2'].tolist(),
        test_df['label3'].tolist(),
        tokenizer
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'测试集大小: {len(test_loader.dataset)}')
    
    # 评估两个模型
    model_paths = [
        'models/bert_multitask_epoch1.pt',
        'models/bert_multitask_epoch2.pt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f'\n====================================')
            print(f'评估模型: {model_path}')
            print(f'====================================')
            evaluate_model(model_path, test_loader, device, num_labels1, num_labels2, num_labels3)
        else:
            print(f'模型文件不存在: {model_path}')

if __name__ == '__main__':
    main()import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

class AIDataset(Dataset):
    def __init__(self, texts, labels1, labels2, labels3, tokenizer, max_length=128):
        self.texts = texts
        self.labels1 = labels1
        self.labels2 = labels2
        self.labels3 = labels3
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]
        label3 = self.labels3[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label1': torch.tensor(label1, dtype=torch.long),
            'label2': torch.tensor(label2, dtype=torch.long),
            'label3': torch.tensor(label3, dtype=torch.long)
        }

class BertMultiTask(nn.Module):
    def __init__(self, num_labels1, num_labels2, num_labels3, model_name='bert-base-chinese'):
        super(BertMultiTask, self).__init__()
        # 加载预训练的Bert模型
        self.bert = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        
        # 获取Bert的隐藏层维度
        self.hidden_size = self.bert.config.hidden_size
        
        # 三个分类头
        self.classifier1 = nn.Linear(self.hidden_size, num_labels1)
        self.classifier2 = nn.Linear(self.hidden_size, num_labels2)
        self.classifier3 = nn.Linear(self.hidden_size, num_labels3)
    
    def forward(self, input_ids, attention_mask):
        # 获取Bert的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS] token的表示作为句子表示
        cls_output = outputs.hidden_states[-1][:, 0, :]
        
        # 三个分类任务的输出
        logits1 = self.classifier1(cls_output)
        logits2 = self.classifier2(cls_output)
        logits3 = self.classifier3(cls_output)
        
        return logits1, logits2, logits3

def evaluate_model(model_path, test_loader, device, num_labels1, num_labels2, num_labels3, le1, le2, le3):
    """评估模型"""
    # 初始化模型
    model = BertMultiTask(num_labels1, num_labels2, num_labels3)
    model.to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 存储预测结果和真实标签
    all_preds1 = []
    all_preds2 = []
    all_preds3 = []
    all_labels1 = []
    all_labels2 = []
    all_labels3 = []
    
    # 评估过程
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='评估中'):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            label3 = batch['label3'].to(device)
            
            # 前向传播
            logits1, logits2, logits3 = model(input_ids, attention_mask)
            
            # 获取预测结果
            preds1 = torch.argmax(logits1, dim=1).cpu().numpy()
            preds2 = torch.argmax(logits2, dim=1).cpu().numpy()
            preds3 = torch.argmax(logits3, dim=1).cpu().numpy()
            
            # 存储结果
            all_preds1.extend(preds1)
            all_preds2.extend(preds2)
            all_preds3.extend(preds3)
            all_labels1.extend(label1.cpu().numpy())
            all_labels2.extend(label2.cpu().numpy())
            all_labels3.extend(label3.cpu().numpy())
    
    # 计算准确率
    accuracy1 = accuracy_score(all_labels1, all_preds1)
    accuracy2 = accuracy_score(all_labels2, all_preds2)
    accuracy3 = accuracy_score(all_labels3, all_preds3)
    
    print(f'\n模型: {os.path.basename(model_path)}')
    print(f'存在AI应用准确率: {accuracy1:.4f}')
    print(f'AI使用方式准确率: {accuracy2:.4f}')
    print(f'AI应用类型准确率: {accuracy3:.4f}')
    print(f'平均准确率: {(accuracy1 + accuracy2 + accuracy3) / 3:.4f}')
    
    return accuracy1, accuracy2, accuracy3

def main():
    # 配置参数
    batch_size = 16
    model_name = 'bert-base-chinese'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载和预处理数据
    print('加载数据...')
    df = pd.read_excel('合并结果.xlsx')
    df = df.fillna('不适用')
    df = df[df['句子'].notna() & (df['句子'] != '')]
    
    # 只使用前1000条数据进行测试
    df = df.head(1000)
    print(f'使用数据量: {len(df)}')
    
    # 编码标签
    print('编码标签...')
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    
    df['label1'] = le1.fit_transform(df['是否存在AI应用'])
    df['label2'] = le2.fit_transform(df['AI使用方式'])
    df['label3'] = le3.fit_transform(df['AI应用类型'])
    
    # 获取标签数量
    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)
    
    print(f'标签数量: 存在AI应用={num_labels1}, AI使用方式={num_labels2}, AI应用类型={num_labels3}')
    
    # 初始化tokenizer
    print('初始化tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 创建测试数据集
    print('创建测试数据集...')
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    test_dataset = AIDataset(
        test_df['句子'].tolist(),
        test_df['label1'].tolist(),
        test_df['label2'].tolist(),
        test_df['label3'].tolist(),
        tokenizer
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f'测试集大小: {len(test_loader.dataset)}')
    
    # 评估两个模型
    model_paths = [
        'models/bert_multitask_epoch1.pt',
        'models/bert_multitask_epoch2.pt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f'\n====================================')
            print(f'评估模型: {model_path}')
            print(f'====================================')
            evaluate_model(model_path, test_loader, device, num_labels1, num_labels2, num_labels3, le1, le2, le3)
        else:
            print(f'模型文件不存在: {model_path}')

if __name__ == '__main__':
    main()