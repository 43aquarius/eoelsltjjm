# API接口文档

## 基础信息

- **基础URL**: `http://localhost:8000/api`
- **请求格式**: JSON (除文件上传外)
- **响应格式**: JSON
- **错误处理**: 统一返回错误码和错误信息

## 接口列表

### 1. 单文本预测

**请求**
- **方法**: POST
- **端点**: `/predict`
- **请求体**:
  ```json
  {
    "text": "公司引入智能生产线，实现生产自动化",
    "model_name": "models/bert_multitask_epoch2.pt"
  }
  ```

**响应**
- **成功**:
  ```json
  {
    "exists_ai": {
      "label": "是",
      "confidence": 0.95
    },
    "usage_method": {
      "label": "产品输出",
      "confidence": 0.88
    },
    "application_type": {
      "label": "决策型",
      "confidence": 0.92
    },
    "timestamp": "2026-04-08T12:00:00Z"
  }
  ```

### 2. 批量预测

**请求**
- **方法**: POST
- **端点**: `/batch-predict`
- **请求体**: `multipart/form-data`
  - `file`: 上传的Excel或CSV文件（必须包含"句子"列）
  - `model_name`: 模型路径

**响应**
- **成功**:
  ```json
  {
    "results": [
      {
        "index": 1,
        "sentence": "公司引入智能生产线，实现生产自动化",
        "exists_ai": "是",
        "usage_method": "产品输出",
        "application_type": "决策型",
        "confidences": [0.95, 0.88, 0.92]
      },
      // 更多结果...
    ],
    "total": 10,
    "processed_time": "2026-04-08T12:00:00Z"
  }
  ```

### 3. 模型评估

**请求**
- **方法**: POST
- **端点**: `/evaluate`
- **请求体**: `multipart/form-data`
  - `file` (可选): 测试数据集文件
  - `model_name`: 模型路径

**响应**
- **成功**:
  ```json
  {
    "model_name": "models/bert_multitask_epoch2.pt",
    "metrics": {
      "exists_ai": {
        "accuracy": 0.92,
        "precision": 0.93,
        "recall": 0.91,
        "f1": 0.92
      },
      "usage_method": {
        "accuracy": 0.88,
        "precision": 0.87,
        "recall": 0.89,
        "f1": 0.88
      },
      "application_type": {
        "accuracy": 0.90,
        "precision": 0.91,
        "recall": 0.89,
        "f1": 0.90
      }
    },
    "confusion_matrices": {
      "exists_ai": [[100, 5], [8, 150]],
      "usage_method": [[80, 10, 5], [12, 90, 8], [5, 7, 70]],
      "application_type": [[75, 10, 5], [8, 85, 7], [6, 8, 80]]
    },
    "sample_count": 263,
    "evaluation_time": "2026-04-08T12:00:00Z"
  }
  ```

### 4. 获取可用模型列表

**请求**
- **方法**: GET
- **端点**: `/models`

**响应**
- **成功**:
  ```json
  [
    {
      "name": "bert_multitask_epoch1.pt",
      "epoch": 1,
      "path": "models/bert_multitask_epoch1.pt"
    },
    {
      "name": "bert_multitask_epoch2.pt",
      "epoch": 2,
      "path": "models/bert_multitask_epoch2.pt"
    }
  ]
  ```

### 5. 获取历史记录

**请求**
- **方法**: GET
- **端点**: `/history`

**响应**
- **成功**:
  ```json
  [
    {
      "id": "record-1",
      "timestamp": "2026-04-08T12:00:00Z",
      "text": "公司引入智能生产线，实现生产自动化",
      "model": "models/bert_multitask_epoch2.pt",
      "results": {
        "exists_ai": {"label": "是", "confidence": 0.95},
        "usage_method": {"label": "产品输出", "confidence": 0.88},
        "application_type": {"label": "决策型", "confidence": 0.92},
        "timestamp": "2026-04-08T12:00:00Z"
      }
    },
    // 更多记录...
  ]
  ```

### 6. 删除历史记录

**请求**
- **方法**: DELETE
- **端点**: `/history/:id`

**响应**
- **成功**:
  ```json
  {
    "status": "success",
    "message": "删除成功"
  }
  ```

### 7. 文件上传

**请求**
- **方法**: POST
- **端点**: `/upload`
- **请求体**: `multipart/form-data`
  - `file`: 要上传的文件

**响应**
- **成功**:
  ```json
  {
    "filename": "uploaded_file.xlsx"
  }
  ```

### 8. 结果下载

**请求**
- **方法**: GET
- **端点**: `/download/:filename`

**响应**
- **成功**: 文件流

## 错误响应

```json
{
  "error": {
    "code": 400,
    "message": "参数错误"
  }
}
```

**错误码说明**
- `400`: 参数错误
- `404`: 资源未找到
- `500`: 服务器内部错误
- `503`: 服务不可用
