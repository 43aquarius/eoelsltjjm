// API接口类型定义

export interface PredictionRequest {
  text: string;
  model_name: string;
}

export interface PredictionResponse {
  exists_ai: { label: string; confidence: number };
  usage_method: { label: string; confidence: number };
  application_type: { label: string; confidence: number };
  timestamp: string;
}

export interface BatchPredictionRequest {
  file: File;
  model_name: string;
}

export interface BatchPredictionResponse {
  results: Array<{
    index: number;
    sentence: string;
    exists_ai: string;
    usage_method: string;
    application_type: string;
    confidences: number[];
  }>;
  total: number;
  processed_time: string;
}

export interface ModelInfo {
  name: string;
  epoch: number;
  path: string;
}

export interface HistoryRecord {
  id: string;
  timestamp: string;
  text: string;
  model: string;
  results: PredictionResponse;
}

export interface EvaluationRequest {
  model_name: string;
  file?: File;
}

export interface EvaluationResponse {
  model_name: string;
  metrics: {
    exists_ai: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
    usage_method: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
    application_type: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
  };
  confusion_matrices: {
    exists_ai: number[][];
    usage_method: number[][];
    application_type: number[][];
  };
  sample_count: number;
  evaluation_time: string;
}
