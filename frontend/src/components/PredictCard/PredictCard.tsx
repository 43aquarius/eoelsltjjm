import React from 'react';
import { Card, Progress, Space, Typography } from 'antd';
import './PredictCard.css';

const { Text, Title } = Typography;

interface PredictCardProps {
  title: string;
  label: string;
  confidence: number;
  color: string;
}

const PredictCard: React.FC<PredictCardProps> = ({ title, label, confidence, color }) => {
  return (
    <Card className="predict-card" bordered={true} style={{ borderColor: color }}>
      <Title level={5} style={{ marginBottom: 16, color: color }}>{title}</Title>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Text strong style={{ fontSize: '16px' }}>{label}</Text>
        <div style={{ marginTop: 8 }}>
          <Progress 
            percent={Math.round(confidence * 100)} 
            status="active" 
            strokeColor={color}
            showInfo={true}
          />
        </div>
        <Text type="secondary" style={{ textAlign: 'right', display: 'block' }}>
          置信度: {Math.round(confidence * 100)}%
        </Text>
      </Space>
    </Card>
  );
};

export default PredictCard;