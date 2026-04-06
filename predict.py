import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import BertTokenizer
from data import load_data, preprocess_data, encode_labels
from model import BertMultiTask

def predict(texts, model_path):
    # 配置参数
    model_name = 'bert-base-chinese'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载标签编码器
    df = load_data('合并结果.xlsx')
    df = preprocess_data(df)
    df, le1, le2, le3 = encode_labels(df)
    
    # 获取标签数量
    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 初始化模型
    model = BertMultiTask(num_labels1, num_labels2, num_labels3, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 预测结果
    results = []
    
    with torch.no_grad():
        for text in texts:
            # 编码文本
            encoding = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 移动数据到设备
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # 前向传播
            logits1, logits2, logits3 = model(input_ids, attention_mask)
            
            # 获取预测结果
            pred1 = torch.argmax(logits1, dim=1).item()
            pred2 = torch.argmax(logits2, dim=1).item()
            pred3 = torch.argmax(logits3, dim=1).item()
            
            # 解码标签
            label1 = le1.inverse_transform([pred1])[0]
            label2 = le2.inverse_transform([pred2])[0]
            label3 = le3.inverse_transform([pred3])[0]
            
            results.append({
                'text': text,
                '是否存在AI应用': label1,
                'AI使用方式': label2,
                'AI应用类型': label3
            })
    
    return results

def main():
    # 示例文本
    sample_texts = [
        '公司引入智能生产线，实现生产自动化',
        '我们使用大数据分析技术优化决策流程',
        '企业采用传统生产方式，未使用人工智能',
        '通过机器学习算法预测市场需求',
        '利用计算机视觉技术进行质量检测'
    ]
    
    # 模型路径
    model_path = 'models/bert_multitask_epoch5.pt'
    
    # 预测
    results = predict(sample_texts, model_path)
    
    # 打印结果
    print('预测结果:')
    for i, result in enumerate(results):
        print(f'\n文本 {i+1}: {result["text"]}')
        print(f'是否存在AI应用: {result["是否存在AI应用"]}')
        print(f'AI使用方式: {result["AI使用方式"]}')
        print(f'AI应用类型: {result["AI应用类型"]}')

if __name__ == '__main__':
    main()
