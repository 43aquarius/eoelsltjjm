import os
import ssl

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import BertTokenizer
from model import BertMultiTask

def test_model(model_path):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 初始化模型
    # 假设标签数量与训练时相同
    num_labels1 = 3  # 存在AI应用
    num_labels2 = 4  # AI使用方式
    num_labels3 = 4  # AI应用类型
    
    model = BertMultiTask(num_labels1, num_labels2, num_labels3)
    model.to(device)
    
    # 加载模型权重
    print(f'加载模型: {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('模型加载成功！')
    
    # 测试预测
    test_texts = [
        "公司使用人工智能技术优化生产流程，提高生产效率。",
        "我们的产品不包含任何AI技术。",
        "公司开发了基于AI的智能决策系统。"
    ]
    
    for i, text in enumerate(test_texts):
        print(f'\n测试文本 {i+1}: {text}')
        
        # 编码文本
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 预测
        with torch.no_grad():
            logits1, logits2, logits3 = model(input_ids, attention_mask)
        
        # 获取预测结果
        pred1 = torch.argmax(logits1, dim=1).item()
        pred2 = torch.argmax(logits2, dim=1).item()
        pred3 = torch.argmax(logits3, dim=1).item()
        
        # 标签映射
        label1_map = {0: '不存在人工智能应用', 1: '存在人工智能应用', 2: '无法判断'}
        label2_map = {0: '不适用', 1: '产品输出', 2: '无法判断', 3: '自身应用'}
        label3_map = {0: '不适用', 1: '决策型', 2: '无法判断', 3: '生产型'}
        
        print(f'预测结果:')
        print(f'存在AI应用: {label1_map[pred1]}')
        print(f'AI使用方式: {label2_map[pred2]}')
        print(f'AI应用类型: {label3_map[pred3]}')

if __name__ == '__main__':
    # 测试两个模型
    model_paths = [
        'models/bert_multitask_epoch1.pt',
        'models/bert_multitask_epoch2.pt'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f'\n====================================')
            print(f'测试模型: {model_path}')
            print(f'====================================')
            test_model(model_path)
        else:
            print(f'模型文件不存在: {model_path}')