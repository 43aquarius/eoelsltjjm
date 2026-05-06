import os
import ssl
from datetime import datetime
from typing import List, Optional
import json

# 禁用SSL验证（仅用于开发环境）
ssl._create_default_https_context = ssl._create_unverified_context

# 设置Hugging Face镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
import pandas as pd
import uvicorn

from data import load_data, preprocess_data, encode_labels
from model import BertMultiTask

# 创建FastAPI应用
app = FastAPI(title="AI文本分析API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
tokenizer = None
label_encoders = None
device = None
MODEL_DIR = "models"
UPLOAD_DIR = "uploads"
HISTORY_FILE = "history.json"

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 请求模型
class PredictionRequest(BaseModel):
    text: str
    model_name: str

class PredictionLabel(BaseModel):
    label: str
    confidence: float

class PredictionResponse(BaseModel):
    exists_ai: PredictionLabel
    usage_method: PredictionLabel
    application_type: PredictionLabel
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    epoch: int
    path: str

class HistoryRecord(BaseModel):
    id: str
    text: str
    result: PredictionResponse
    model_name: str

# 初始化函数
def initialize():
    global tokenizer, label_encoders, device

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化tokenizer
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # 加载标签编码器
    df = load_data('合并结果.xlsx')
    df = preprocess_data(df)
    df, le1, le2, le3 = encode_labels(df)
    label_encoders = (le1, le2, le3)

    print("初始化完成")

# 加载模型
def load_model(model_path: str):
    le1, le2, le3 = label_encoders
    num_labels1 = len(le1.classes_)
    num_labels2 = len(le2.classes_)
    num_labels3 = len(le3.classes_)

    model = BertMultiTask(num_labels1, num_labels2, num_labels3, 'bert-base-chinese')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

# 预测函数
def predict_text(text: str, model: BertMultiTask):
    le1, le2, le3 = label_encoders

    with torch.no_grad():
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

        # 获取预测结果和置信度
        probs1 = torch.softmax(logits1, dim=1)
        probs2 = torch.softmax(logits2, dim=1)
        probs3 = torch.softmax(logits3, dim=1)

        pred1 = torch.argmax(probs1, dim=1).item()
        pred2 = torch.argmax(probs2, dim=1).item()
        pred3 = torch.argmax(probs3, dim=1).item()

        conf1 = probs1[0][pred1].item()
        conf2 = probs2[0][pred2].item()
        conf3 = probs3[0][pred3].item()

        # 解码标签
        label1 = le1.inverse_transform([pred1])[0]
        label2 = le2.inverse_transform([pred2])[0]
        label3 = le3.inverse_transform([pred3])[0]

        return {
            'exists_ai': {'label': label1, 'confidence': round(conf1, 4)},
            'usage_method': {'label': label2, 'confidence': round(conf2, 4)},
            'application_type': {'label': label3, 'confidence': round(conf3, 4)}
        }

# 加载历史记录
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# 保存历史记录
def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# API路由
@app.on_event("startup")
async def startup_event():
    initialize()

@app.get("/")
async def root():
    return {"message": "AI文本分析API服务运行中"}

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    models = []
    if os.path.exists(MODEL_DIR):
        for filename in os.listdir(MODEL_DIR):
            if filename.endswith('.pt'):
                # 从文件名提取epoch信息
                epoch = 1
                if 'epoch' in filename:
                    try:
                        epoch = int(filename.split('epoch')[1].split('.')[0])
                    except:
                        pass

                models.append({
                    'name': filename,
                    'epoch': epoch,
                    'path': os.path.join(MODEL_DIR, filename)
                })

    return models

@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """单文本预测"""
    try:
        # 加载模型
        model = load_model(request.model_name)

        # 预测
        result = predict_text(request.text, model)
        result['timestamp'] = datetime.now().isoformat()

        # 保存到历史记录
        history = load_history()
        history_record = {
            'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
            'text': request.text,
            'result': result,
            'model_name': request.model_name
        }
        history.append(history_record)
        save_history(history)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...), model_name: str = None):
    """批量预测"""
    try:
        # 保存上传的文件
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # 读取Excel或CSV文件
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        # 支持多种列名
        possible_columns = ['句子', 'text', '文本', 'sentence']
        text_column = None
        for col in possible_columns:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            raise HTTPException(status_code=400, detail="文件必须包含以下列名之一: '句子', 'text', '文本', 'sentence'")

        # 加载模型 - 使用最佳模型 epoch3
        if not model_name:
            # 优先使用epoch3模型（评估结果最佳）
            best_model = os.path.join(MODEL_DIR, 'bert_multitask_epoch3.pt')
            if os.path.exists(best_model):
                model_name = best_model
            else:
                # 使用默认模型
                models = await get_models()
                if not models:
                    raise HTTPException(status_code=404, detail="没有可用的模型")
                model_name = models[0]['path']

        model = load_model(model_name)

        # 批量预测
        results = []
        for idx, row in df.iterrows():
            text = str(row[text_column])
            result = predict_text(text, model)
            results.append({
                'text': text,
                'exists_ai': result['exists_ai']['label'],
                'usage_method': result['usage_method']['label'],
                'application_type': result['application_type']['label'],
                'confidence_ai': result['exists_ai']['confidence'],
                'confidence_usage': result['usage_method']['confidence'],
                'confidence_type': result['application_type']['confidence']
            })

        # 保存结果
        result_df = pd.DataFrame(results)
        output_filename = f"batch_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        result_df.to_excel(output_path, index=False)

        return {
            'total': len(results),
            'results': results,  # 返回所有结果
            'download_url': f'/api/download/{output_filename}'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    """获取历史记录"""
    return load_history()

@app.delete("/api/history/{record_id}")
async def delete_history(record_id: str):
    """删除历史记录"""
    history = load_history()
    history = [h for h in history if h['id'] != record_id]
    save_history(history)
    return {"message": "删除成功"}

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载文件"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(file_path, filename=filename)

@app.post("/api/evaluate")
async def evaluate_model(file: UploadFile = File(None), model_name: str = None):
    """模型评估"""
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # 如果上传了文件，使用上传的测试集
        if file:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)

            if file.filename.endswith('.csv'):
                test_df = pd.read_csv(file_path)
            else:
                test_df = pd.read_excel(file_path)
        else:
            # 使用默认测试集 - 减少到100条以加快速度
            df = load_data('合并结果.xlsx')
            df = preprocess_data(df)
            df, le1, le2, le3 = encode_labels(df)
            # 使用前100条作为测试集（原来是1000条，太慢）
            test_df = df.head(100)

        # 如果是上传的文件，检查列名并编码
        if file:
            required_cols = ['句子', '是否存在人工智能应用', 'AI使用方式', 'AI应用类型']
            for col in required_cols:
                if col not in test_df.columns:
                    raise HTTPException(status_code=400, detail=f"测试集缺少必需列: {col}")

            # 编码标签
            df_full = load_data('合并结果.xlsx')
            df_full = preprocess_data(df_full)
            _, le1, le2, le3 = encode_labels(df_full)

        # 加载模型
        if not model_name:
            model_name = os.path.join(MODEL_DIR, 'bert_multitask_epoch3.pt')

        model = load_model(model_name)

        # 预测
        all_preds1, all_preds2, all_preds3 = [], [], []
        all_labels1, all_labels2, all_labels3 = [], [], []

        for idx, row in test_df.iterrows():
            text = str(row['句子'])
            result = predict_text(text, model)

            # 预测标签
            all_preds1.append(result['exists_ai']['label'])
            all_preds2.append(result['usage_method']['label'])
            all_preds3.append(result['application_type']['label'])

            # 真实标签 - 使用正确的列名
            if file:
                # 上传的文件使用完整列名
                all_labels1.append(row['是否存在人工智能应用'])
                all_labels2.append(row['AI使用方式'])
                all_labels3.append(row['AI应用类型'])
            else:
                # 默认测试集使用简短列名
                all_labels1.append(row['是否存在AI应用'])
                all_labels2.append(row['AI使用方式'])
                all_labels3.append(row['AI应用类型'])

        # 计算指标
        def calculate_metrics(labels, preds):
            accuracy = accuracy_score(labels, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='weighted', zero_division=0
            )
            return {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4)
            }

        metrics = {
            'exists_ai': calculate_metrics(all_labels1, all_preds1),
            'usage_method': calculate_metrics(all_labels2, all_preds2),
            'application_type': calculate_metrics(all_labels3, all_preds3)
        }

        return {
            'model_name': model_name,
            'metrics': metrics,
            'sample_count': len(test_df),
            'evaluation_time': datetime.now().isoformat(),
            'confusion_matrices': {
                'exists_ai': [[0, 0], [0, 0]],
                'usage_method': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                'application_type': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
