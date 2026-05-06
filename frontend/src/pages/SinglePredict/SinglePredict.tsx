import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Select, Space, Typography, message, Row, Col, Spin } from 'antd';
import { apiService } from '../../services/api';
import PredictCard from '../../components/PredictCard/PredictCard';
import { ModelInfo, PredictionRequest, PredictionResponse } from '../../types';
import './SinglePredict.css';

const { TextArea } = Input;
const { Title, Text } = Typography;

const SinglePredict: React.FC = () => {
  const [form] = Form.useForm();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);

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

  // 处理预测
  const handlePredict = async (values: any) => {
    setPredicting(true);
    try {
      const request: PredictionRequest = {
        text: values.text,
        model_name: values.model
      };

      const response = await apiService.predict(request);

      setResult(response);
      message.success('预测成功');
    } catch (error) {
      message.error('预测失败，请重试');
      console.error('Error predicting:', error);
    } finally {
      setPredicting(false);
    }
  };

  // 清空表单
  const handleReset = () => {
    form.resetFields();
    setResult(null);
  };

  return (
    <div className="single-predict">
      <Title level={3}>单文本预测</Title>
      <Text type="secondary">输入文本，选择模型，点击开始分析按钮进行预测</Text>

      <Form
        form={form}
        layout="vertical"
        onFinish={handlePredict}
        style={{ marginTop: 24 }}
      >
        <Form.Item
          name="text"
          label="输入文本"
          rules={[
            { required: true, message: '请输入文本' },
            { max: 512, message: '文本长度不能超过512字符' }
          ]}
        >
          <TextArea
            rows={6}
            placeholder="请输入需要分析的文本..."
            maxLength={512}
            showCount
          />
        </Form.Item>

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

        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={predicting}>
              开始分析
            </Button>
            <Button onClick={handleReset} disabled={predicting}>
              清空
            </Button>
          </Space>
        </Form.Item>
      </Form>

      {predicting && (
        <div className="loading-overlay">
          <Spin size="large" tip="分析中..." />
        </div>
      )}

      {result && (
        <div className="result-section">
          <Title level={4}>预测结果</Title>
          <Text type="secondary">预测时间: {new Date(result.timestamp).toLocaleString()}</Text>
          
          <Row gutter={16} style={{ marginTop: 24 }}>
            <Col span={8}>
              <PredictCard
                title="AI应用存在性"
                label={result.exists_ai.label}
                confidence={result.exists_ai.confidence}
                color="#1890ff"
              />
            </Col>
            <Col span={8}>
              <PredictCard
                title="AI使用方式"
                label={result.usage_method.label}
                confidence={result.usage_method.confidence}
                color="#52c41a"
              />
            </Col>
            <Col span={8}>
              <PredictCard
                title="AI应用类型"
                label={result.application_type.label}
                confidence={result.application_type.confidence}
                color="#faad14"
              />
            </Col>
          </Row>
        </div>
      )}
    </div>
  );
};

export default SinglePredict;