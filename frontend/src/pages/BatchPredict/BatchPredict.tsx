import React, { useState } from 'react';
import { Form, Button, Table, Space, Typography, message, Spin } from 'antd';
import { DownloadOutlined } from '@ant-design/icons';
import FileUpload from '../../components/FileUpload/FileUpload';
import { apiService } from '../../services/api';
import './BatchPredict.css';

const { Title, Text } = Typography;

interface BatchResult {
  key: string;
  index: number;
  sentence: string;
  exists_ai: string;
  usage_method: string;
  application_type: string;
  confidence: number;
}

const BatchPredict: React.FC = () => {
  const [form] = Form.useForm();
  const [predicting, setPredicting] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [total, setTotal] = useState(0);
  const [processedTime, setProcessedTime] = useState('');
  const [downloadUrl, setDownloadUrl] = useState<string>('');

  // 处理文件上传
  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    message.success('文件上传成功');
  };

  // 处理批量预测
  const handleBatchPredict = async (values: any) => {
    if (!file) {
      message.error('请先上传文件');
      return;
    }

    setPredicting(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await apiService.batchPredict(formData);

      // 转换响应数据为表格格式
      const batchResults: BatchResult[] = response.results.map((item: any, index: number) => ({
        key: `${index}`,
        index: index + 1,
        sentence: item.text,
        exists_ai: item.exists_ai,
        usage_method: item.usage_method,
        application_type: item.application_type,
        confidence: (item.confidence_ai + item.confidence_usage + item.confidence_type) / 3
      }));

      setResults(batchResults);
      setTotal(response.total);
      setDownloadUrl(response.download_url);
      setProcessedTime(new Date().toLocaleString());
      message.success('批量预测成功');
    } catch (error) {
      message.error('批量预测失败，请重试');
      console.error('Error batch predicting:', error);
    } finally {
      setPredicting(false);
    }
  };

  // 处理结果导出
  const handleExport = async () => {
    if (results.length === 0) {
      message.error('没有可导出的结果');
      return;
    }

    if (!downloadUrl) {
      message.error('下载链接不存在');
      return;
    }

    try {
      // 提取文件名
      const filename = downloadUrl.split('/').pop() || 'batch_result.xlsx';
      const blob = await apiService.downloadFile(filename);

      // 创建下载链接
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      message.success('导出成功');
    } catch (error) {
      message.error('导出失败，请重试');
      console.error('Error exporting:', error);
    }
  };

  // 表格列定义
  const columns = [
    {
      title: '序号',
      dataIndex: 'index',
      key: 'index',
      width: 80
    },
    {
      title: '原始句子',
      dataIndex: 'sentence',
      key: 'sentence',
      ellipsis: true
    },
    {
      title: 'AI存在性',
      dataIndex: 'exists_ai',
      key: 'exists_ai',
      width: 100
    },
    {
      title: '使用方式',
      dataIndex: 'usage_method',
      key: 'usage_method',
      width: 120
    },
    {
      title: '应用类型',
      dataIndex: 'application_type',
      key: 'application_type',
      width: 120
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      render: (confidence: number) => `${Math.round(confidence * 100)}%`
    }
  ];

  return (
    <div className="batch-predict">
      <Title level={3}>批量预测</Title>
      <Text type="secondary">上传包含"句子"列的Excel或CSV文件进行批量分析（自动使用最佳模型）</Text>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleBatchPredict}
        style={{ marginTop: 24 }}
      >
        <Form.Item label="上传文件">
          <FileUpload
            accept=".xlsx,.csv"
            maxSize={10 * 1024 * 1024}
            onFileUpload={handleFileUpload}
            disabled={predicting}
            requiredColumns={['句子']}
            description="上传包含待分析文本的Excel或CSV文件。列名可以是：句子、text、文本、sentence"
          />
        </Form.Item>

        <Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={predicting}>
              开始批量分析
            </Button>
          </Space>
        </Form.Item>
      </Form>

      {predicting && (
        <div className="loading-overlay">
          <Spin size="large" tip="批量分析中..." />
        </div>
      )}

      {results.length > 0 && (
        <div className="result-section">
          <div className="result-header">
            <Title level={4}>预测结果</Title>
            <Space>
              <Text type="secondary">
                处理时间: {processedTime}
              </Text>
              <Text type="secondary">
                总条数: {total}
              </Text>
              <Button 
                type="primary" 
                icon={<DownloadOutlined />} 
                onClick={handleExport}
              >
                导出结果
              </Button>
            </Space>
          </div>

          <Table
            columns={columns}
            dataSource={results}
            pagination={{
              pageSize: 20,
              showSizeChanger: true,
              showQuickJumper: true
            }}
            scroll={{ x: 800 }}
          />
        </div>
      )}
    </div>
  );
};

export default BatchPredict;