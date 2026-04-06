import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch

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

def load_data(file_path):
    """加载数据"""
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    """预处理数据"""
    # 处理缺失值
    df = df.fillna('不适用')
    
    # 过滤无效数据
    df = df[df['句子'].notna() & (df['句子'] != '')]
    
    return df

def encode_labels(df):
    """编码标签"""
    # 初始化标签编码器
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    
    # 编码标签
    df['label1'] = le1.fit_transform(df['是否存在AI应用'])
    df['label2'] = le2.fit_transform(df['AI使用方式'])
    df['label3'] = le3.fit_transform(df['AI应用类型'])
    
    return df, le1, le2, le3

def create_data_loaders(df, tokenizer, batch_size=32, test_size=0.2):
    """创建数据加载器"""
    from sklearn.model_selection import train_test_split
    
    # 分割数据
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # 创建数据集
    train_dataset = AIDataset(
        train_df['句子'].tolist(),
        train_df['label1'].tolist(),
        train_df['label2'].tolist(),
        train_df['label3'].tolist(),
        tokenizer
    )
    
    test_dataset = AIDataset(
        test_df['句子'].tolist(),
        test_df['label1'].tolist(),
        test_df['label2'].tolist(),
        test_df['label3'].tolist(),
        tokenizer
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
