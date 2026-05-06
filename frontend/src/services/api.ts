import axios from 'axios';
import { PredictionRequest, PredictionResponse, BatchPredictionResponse, ModelInfo, HistoryRecord, EvaluationRequest, EvaluationResponse } from '../types';

// 基础URL
const BASE_URL = '/api';

// 创建axios实例
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 180000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// API接口
export const apiService = {
  // 单文本预测
  predict: async (data: PredictionRequest): Promise<PredictionResponse> => {
    const response = await api.post('/predict', data);
    return response.data;
  },

  // 批量预测
  batchPredict: async (formData: FormData): Promise<BatchPredictionResponse> => {
    const response = await api.post('/batch-predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // 模型评估
  evaluate: async (formData: FormData): Promise<EvaluationResponse> => {
    const response = await api.post('/evaluate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // 获取可用模型列表
  getModels: async (): Promise<ModelInfo[]> => {
    const response = await api.get('/models');
    return response.data;
  },

  // 获取历史记录
  getHistory: async (): Promise<HistoryRecord[]> => {
    const response = await api.get('/history');
    return response.data;
  },

  // 删除历史记录
  deleteHistory: async (id: string): Promise<void> => {
    await api.delete(`/history/${id}`);
  },

  // 上传文件
  uploadFile: async (file: File): Promise<{ filename: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // 下载文件
  downloadFile: async (filename: string): Promise<Blob> => {
    const response = await api.get(`/download/${filename}`, {
      responseType: 'blob'
    });
    return response.data;
  }
};
