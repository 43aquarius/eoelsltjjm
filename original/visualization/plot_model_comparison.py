import os
import ssl
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import sys
sys.path.append('..')
from data import load_data, preprocess_data, encode_labels, create_data_loaders
from model import BertMultiTask

def evaluate_model(model_path):
    """评估单个模型并返回指标"""
    batch_size = 16
    model_name = 'bert-base-chinese'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    print('加载数据...')
    df = load_data('../合并结果.xlsx')
    df = preprocess_data(df)
    df, le1, le2, le3 = encode_labels(df)

    df = df.head(50000)
    print(f'使用数据量: {len(df)}')

    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    _, test_loader = create_data_loaders(df, tokenizer, batch_size=batch_size)

    model = BertMultiTask(num_labels1, num_labels2, num_labels3, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_labels1, all_preds1 = [], []
    all_labels2, all_preds2 = [], []
    all_labels3, all_preds3 = [], []

    print(f'评估模型: {model_path}')
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label1 = batch['label1'].to(device)
            label2 = batch['label2'].to(device)
            label3 = batch['label3'].to(device)

            logits1, logits2, logits3 = model(input_ids, attention_mask)

            preds1 = torch.argmax(logits1, dim=1)
            preds2 = torch.argmax(logits2, dim=1)
            preds3 = torch.argmax(logits3, dim=1)

            all_labels1.extend(label1.cpu().numpy())
            all_preds1.extend(preds1.cpu().numpy())
            all_labels2.extend(label2.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_labels3.extend(label3.cpu().numpy())
            all_preds3.extend(preds3.cpu().numpy())

    def calculate_metrics(labels, preds):
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    metrics = {
        'task1': calculate_metrics(all_labels1, all_preds1),
        'task2': calculate_metrics(all_labels2, all_preds2),
        'task3': calculate_metrics(all_labels3, all_preds3),
        'labels': {
            'task1': le1.classes_.tolist(),
            'task2': le2.classes_.tolist(),
            'task3': le3.classes_.tolist()
        },
        'predictions': {
            'task1': {'labels': all_labels1, 'preds': all_preds1},
            'task2': {'labels': all_labels2, 'preds': all_preds2},
            'task3': {'labels': all_labels3, 'preds': all_preds3}
        }
    }

    return metrics

def plot_metrics_comparison(all_metrics, save_dir='plots'):
    """绘制四个模型的指标对比图"""
    os.makedirs(save_dir, exist_ok=True)

    model_names = ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4']
    tasks = ['task1', 'task2', 'task3']
    task_names = ['任务1: 是否存在AI应用', '任务2: AI使用方式', '任务3: AI应用类型']
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']

    # 1. 绘制每个任务的四个指标对比图
    for task_idx, (task, task_name) in enumerate(zip(tasks, task_names)):
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(model_names))
        width = 0.2

        for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
            values = [all_metrics[j][task][metric] for j in range(4)]
            ax.bar(x + i * width, values, width, label=label)

        ax.set_xlabel('模型', fontsize=12)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title(f'{task_name} - 各指标对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(f'{save_dir}/{task}_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'已保存: {save_dir}/{task}_metrics_comparison.png')

    # 2. 绘制所有任务的准确率对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.25

    for i, (task, task_name) in enumerate(zip(tasks, task_names)):
        values = [all_metrics[j][task]['accuracy'] for j in range(4)]
        ax.bar(x + i * width, values, width, label=task_name)

    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('四个模型在三个任务上的准确率对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_tasks_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {save_dir}/all_tasks_accuracy_comparison.png')

    # 3. 绘制F1分数对比图
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (task, task_name) in enumerate(zip(tasks, task_names)):
        values = [all_metrics[j][task]['f1'] for j in range(4)]
        ax.bar(x + i * width, values, width, label=task_name)

    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('F1分数', fontsize=12)
    ax.set_title('四个模型在三个任务上的F1分数对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_tasks_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {save_dir}/all_tasks_f1_comparison.png')

    # 4. 绘制每个模型的雷达图
    for model_idx, model_name in enumerate(model_names):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

        for task_idx, (task, task_name, ax) in enumerate(zip(tasks, task_names, axes)):
            categories = metrics_labels
            values = [all_metrics[model_idx][task][metric] for metric in metrics_names]
            values += values[:1]  # 闭合图形

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(task_name, fontsize=12, fontweight='bold', pad=20)
            ax.grid(True)

        plt.suptitle(f'{model_name} - 三个任务的性能雷达图', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/radar_model_{model_idx+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'已保存: {save_dir}/radar_model_{model_idx+1}.png')

    # 5. 绘制训练进度趋势图（四个epoch的变化）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for metric_idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        ax = axes[metric_idx]

        for task, task_name in zip(tasks, task_names):
            values = [all_metrics[j][task][metric] for j in range(4)]
            ax.plot(range(1, 5), values, marker='o', linewidth=2, markersize=8, label=task_name)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}随训练轮次的变化', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 5))
        ax.set_ylim([0, 1.0])

    plt.suptitle('模型性能随训练轮次的变化趋势', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存: {save_dir}/training_progress.png')

def main():
    model_paths = [
        '../models/bert_multitask_epoch1.pt',
        '../models/bert_multitask_epoch2.pt',
        '../models/bert_multitask_epoch3.pt',
        '../models/bert_multitask_epoch4.pt'
    ]

    all_metrics = []

    # 评估所有模型
    for model_path in model_paths:
        if os.path.exists(model_path):
            metrics = evaluate_model(model_path)
            all_metrics.append(metrics)
        else:
            print(f'模型文件不存在: {model_path}')
            return

    # 保存评估结果
    results_to_save = []
    for i, metrics in enumerate(all_metrics):
        result = {
            'model': f'epoch{i+1}',
            'task1': {k: v for k, v in metrics['task1'].items()},
            'task2': {k: v for k, v in metrics['task2'].items()},
            'task3': {k: v for k, v in metrics['task3'].items()}
        }
        results_to_save.append(result)

    with open('plots/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    print('已保存评估结果: plots/evaluation_results.json')

    # 绘制对比图
    plot_metrics_comparison(all_metrics)

    print('\n所有图表已生成完成！')

if __name__ == '__main__':
    main()
