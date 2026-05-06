import React, { useState, useEffect } from 'react';
import { Table, Button, Space, Typography, message, DatePicker, Input, Modal, notification } from 'antd';
import { DeleteOutlined, DownloadOutlined, EyeOutlined } from '@ant-design/icons';
import { apiService } from '../../services/api';
import { HistoryRecord } from '../../types';
import './History.css';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Search } = Input;

const History: React.FC = () => {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<HistoryRecord | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [dateRange, setDateRange] = useState<[Date, Date] | null>(null);

  // 加载历史记录
  useEffect(() => {
    const loadHistory = async () => {
      setLoading(true);
      try {
        const data = await apiService.getHistory();
        // 转换数据格式以匹配前端类型
        const formattedData: HistoryRecord[] = data.map((item: any) => ({
          id: item.id,
          timestamp: item.result.timestamp,
          text: item.text,
          model: item.model_name,
          results: item.result
        }));
        setRecords(formattedData);
      } catch (error) {
        message.error('加载历史记录失败');
        console.error('Error loading history:', error);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  // 处理查看详情
  const handleViewDetail = (record: HistoryRecord) => {
    setSelectedRecord(record);
    setModalVisible(true);
  };

  // 处理删除记录
  const handleDelete = async (id: string) => {
    try {
      await apiService.deleteHistory(id);
      setRecords(records.filter(record => record.id !== id));
      message.success('删除成功');
    } catch (error) {
      message.error('删除失败，请重试');
      console.error('Error deleting record:', error);
    }
  };

  // 处理批量删除
  const handleBatchDelete = async () => {
    if (records.length === 0) {
      message.warning('没有可删除的记录');
      return;
    }

    Modal.confirm({
      title: '确认删除',
      content: '确定要删除所有历史记录吗？此操作不可恢复。',
      okText: '确定',
      cancelText: '取消',
      onOk: async () => {
        try {
          // 删除所有记录
          for (const record of records) {
            await apiService.deleteHistory(record.id);
          }
          setRecords([]);
          message.success('批量删除成功');
        } catch (error) {
          message.error('批量删除失败，请重试');
          console.error('Error batch deleting:', error);
        }
      }
    });
  };

  // 处理导出记录
  const handleExport = async () => {
    if (records.length === 0) {
      message.warning('没有可导出的记录');
      return;
    }

    try {
      // 将记录转换为Excel格式
      const exportData = records.map((record, index) => ({
        '序号': index + 1,
        '时间': new Date(record.timestamp).toLocaleString(),
        '输入文本': record.text,
        '使用模型': record.model,
        'AI应用存在性': record.results.exists_ai.label,
        'AI应用存在性置信度': `${Math.round(record.results.exists_ai.confidence * 100)}%`,
        'AI使用方式': record.results.usage_method.label,
        'AI使用方式置信度': `${Math.round(record.results.usage_method.confidence * 100)}%`,
        'AI应用类型': record.results.application_type.label,
        'AI应用类型置信度': `${Math.round(record.results.application_type.confidence * 100)}%`
      }));

      // 创建CSV内容
      const headers = Object.keys(exportData[0]);
      const csvContent = [
        headers.join(','),
        ...exportData.map(row => headers.map(header => `"${row[header as keyof typeof row]}"`).join(','))
      ].join('\n');

      // 创建Blob并下载
      const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `历史记录_${new Date().toISOString().slice(0, 10)}.csv`;
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
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleString()
    },
    {
      title: '输入文本',
      dataIndex: 'text',
      key: 'text',
      ellipsis: true
    },
    {
      title: '使用模型',
      dataIndex: 'model',
      key: 'model'
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: HistoryRecord) => (
        <Space size="middle">
          <Button 
            type="text" 
            icon={<EyeOutlined />} 
            onClick={() => handleViewDetail(record)}
          >
            查看
          </Button>
          <Button 
            type="text" 
            danger 
            icon={<DeleteOutlined />} 
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div className="history">
      <Title level={3}>历史记录</Title>
      <Text type="secondary">查看和管理历史预测记录</Text>

      <div className="history-header" style={{ marginTop: 24, marginBottom: 16 }}>
        <Space>
          <Search
            placeholder="搜索文本内容"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: 300 }}
          />
          <RangePicker
            value={dateRange}
            onChange={(dates) => setDateRange(dates as [Date, Date] | null)}
          />
          <Button 
            type="primary" 
            icon={<DownloadOutlined />} 
            onClick={handleExport}
          >
            导出记录
          </Button>
          <Button 
            danger 
            onClick={handleBatchDelete}
          >
            批量删除
          </Button>
        </Space>
      </div>

      <Table
        columns={columns}
        dataSource={records}
        rowKey="id"
        loading={loading}
        pagination={{
          pageSize: 15,
          showSizeChanger: true,
          showQuickJumper: true
        }}
      />

      <Modal
        title="预测详情"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            关闭
          </Button>
        ]}
      >
        {selectedRecord && (
          <div className="modal-content">
            <Text strong>输入文本:</Text>
            <p style={{ margin: '8px 0 16px 0' }}>{selectedRecord.text}</p>

            <Text strong>使用模型:</Text>
            <p style={{ margin: '8px 0 16px 0' }}>{selectedRecord.model}</p>

            <Text strong>预测时间:</Text>
            <p style={{ margin: '8px 0 16px 0' }}>
              {new Date(selectedRecord.timestamp).toLocaleString()}
            </p>

            <Text strong>预测结果:</Text>
            <div style={{ margin: '8px 0', padding: '16px', background: '#fafafa', borderRadius: '4px' }}>
              <p>AI应用存在性: {selectedRecord.results.exists_ai.label} (置信度: {Math.round(selectedRecord.results.exists_ai.confidence * 100)}%)</p>
              <p>AI使用方式: {selectedRecord.results.usage_method.label} (置信度: {Math.round(selectedRecord.results.usage_method.confidence * 100)}%)</p>
              <p>AI应用类型: {selectedRecord.results.application_type.label} (置信度: {Math.round(selectedRecord.results.application_type.confidence * 100)}%)</p>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default History;