import React, { useState, useEffect } from 'react';
import { Form, Select, Button, Space, Typography, message, Spin, Card, Row, Col, Divider } from 'antd';
import FileUpload from '../../components/FileUpload/FileUpload';
import MetricsChart from '../../components/Charts/MetricsChart';
import { apiService } from '../../services/api';
import { ModelInfo, EvaluationResponse } from '../../types';
import './ModelEvaluation.css';

const { Title, Text } = Typography;

const ModelEvaluation: React.FC = () => {
  const [form] = Form.useForm();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResponse | null>(null);

  // 加载模型列表
  useEffect(() => {
    const loadModels = async () => {
      setLoading(true);
      try {
        const data = await apiService.getModels();
        setModels(data);
      } catch (error) {
        message.error('加载模型列表失败');
        console.error('Error loading models:', error);
      } finally {
        setLoading(false);
      }
    };

    loadModels();
  }, []);

  // 处理文件上传
  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    message.success('文件上传成功');
  };

  // 处理模型评估
  const handleEvaluate = async (values: any) => {
    setEvaluating(true);
    try {
      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      }
      formData.append('model_name', values.model);

      const response = await apiService.evaluate(formData);

      setEvaluationResult(response);
      message.success('模型评估成功');
    } catch (error) {
      message.error('模型评估失败，请重试');
      console.error('Error evaluating model:', error);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="model-evaluation">
      <Title level={3}>模型评估</Title>
      <Text type="secondary">选择模型和测试数据集，评估模型性能</Text>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleEvaluate}
        style={{ marginTop: 24 }}
      >
        <Form.Item
          name="model"
          label="选择模型"
          rules={[{ required: true, message: '请选择模型' }]}
        >
          <Select placeholder="请选择模型">
            {models.map((model) => (
              <Select.Option key={model.path} value={model.path}>
                {model.name} (Epoch {model.epoch})
              </Select.Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item label="测试数据集（可选）">
          <FileUpload
            accept=".xlsx,.csv"
            maxSize={10 * 1024 * 1024}
            onFileUpload={handleFileUpload}
            disabled={evaluating}
            requiredColumns={['句子', '是否存在人工智能应用', 'AI使用方式', 'AI应用类型']}
            description="上传包含真实标签的测试数据集。不上传则使用默认测试集（1000条数据）"
          />
          <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
            不上传则使用默认测试集
          </Text>
        </Form.Item>

        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={evaluating}>
              开始评估
            </Button>
          </Space>
        </Form.Item>
      </Form>

      {evaluating && (
        <div className="loading-overlay">
          <Spin size="large" tip="评估中..." />
        </div>
      )}

      {evaluationResult && (
        <div className="result-section">
          <Title level={4}>评估结果</Title>
          <Text type="secondary">
            评估时间: {new Date(evaluationResult.evaluation_time).toLocaleString()}
          </Text>
          <Text type="secondary" style={{ marginLeft: 24 }}>
            样本数量: {evaluationResult.sample_count}
          </Text>

          <Divider />

          <Row gutter={16} style={{ marginTop: 24 }}>
            <Col span={8}>
              <Card title="AI应用存在性" bordered={true} style={{ borderColor: '#1890ff' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>准确率: {Math.round(evaluationResult.metrics.exists_ai.accuracy * 100)}%</Text>
                  <Text>精确率: {Math.round(evaluationResult.metrics.exists_ai.precision * 100)}%</Text>
                  <Text>召回率: {Math.round(evaluationResult.metrics.exists_ai.recall * 100)}%</Text>
                  <Text>F1分数: {Math.round(evaluationResult.metrics.exists_ai.f1 * 100)}%</Text>
                </Space>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="AI使用方式" bordered={true} style={{ borderColor: '#52c41a' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>准确率: {Math.round(evaluationResult.metrics.usage_method.accuracy * 100)}%</Text>
                  <Text>精确率: {Math.round(evaluationResult.metrics.usage_method.precision * 100)}%</Text>
                  <Text>召回率: {Math.round(evaluationResult.metrics.usage_method.recall * 100)}%</Text>
                  <Text>F1分数: {Math.round(evaluationResult.metrics.usage_method.f1 * 100)}%</Text>
                </Space>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="AI应用类型" bordered={true} style={{ borderColor: '#faad14' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>准确率: {Math.round(evaluationResult.metrics.application_type.accuracy * 100)}%</Text>
                  <Text>精确率: {Math.round(evaluationResult.metrics.application_type.precision * 100)}%</Text>
                  <Text>召回率: {Math.round(evaluationResult.metrics.application_type.recall * 100)}%</Text>
                  <Text>F1分数: {Math.round(evaluationResult.metrics.application_type.f1 * 100)}%</Text>
                </Space>
              </Card>
            </Col>
          </Row>

          <MetricsChart data={evaluationResult.metrics} />
        </div>
      )}
    </div>
  );
};

export default ModelEvaluation;