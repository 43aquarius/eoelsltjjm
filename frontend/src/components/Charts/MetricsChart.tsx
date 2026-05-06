import React from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import './Charts.css';

interface MetricsChartProps {
  data: {
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
}

const MetricsChart: React.FC<MetricsChartProps> = ({ data }) => {
  // 雷达图数据
  const radarData = [
    {
      subject: '准确率',
      'AI存在性': data.exists_ai.accuracy * 100,
      '使用方式': data.usage_method.accuracy * 100,
      '应用类型': data.application_type.accuracy * 100,
    },
    {
      subject: '精确率',
      'AI存在性': data.exists_ai.precision * 100,
      '使用方式': data.usage_method.precision * 100,
      '应用类型': data.application_type.precision * 100,
    },
    {
      subject: '召回率',
      'AI存在性': data.exists_ai.recall * 100,
      '使用方式': data.usage_method.recall * 100,
      '应用类型': data.application_type.recall * 100,
    },
    {
      subject: 'F1分数',
      'AI存在性': data.exists_ai.f1 * 100,
      '使用方式': data.usage_method.f1 * 100,
      '应用类型': data.application_type.f1 * 100,
    },
  ];

  // 柱状图数据
  const barData = [
    {
      name: 'AI存在性',
      准确率: data.exists_ai.accuracy * 100,
      精确率: data.exists_ai.precision * 100,
      召回率: data.exists_ai.recall * 100,
      F1分数: data.exists_ai.f1 * 100,
    },
    {
      name: '使用方式',
      准确率: data.usage_method.accuracy * 100,
      精确率: data.usage_method.precision * 100,
      召回率: data.usage_method.recall * 100,
      F1分数: data.usage_method.f1 * 100,
    },
    {
      name: '应用类型',
      准确率: data.application_type.accuracy * 100,
      精确率: data.application_type.precision * 100,
      召回率: data.application_type.recall * 100,
      F1分数: data.application_type.f1 * 100,
    },
  ];

  return (
    <div className="metrics-chart">
      <div className="chart-container">
        <h3>指标雷达图</h3>
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart outerRadius={150} data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" />
            <PolarRadiusAxis angle={30} domain={[0, 100]} />
            <Radar name="AI存在性" dataKey="AI存在性" stroke="#1890ff" fill="#1890ff" fillOpacity={0.5} />
            <Radar name="使用方式" dataKey="使用方式" stroke="#52c41a" fill="#52c41a" fillOpacity={0.5} />
            <Radar name="应用类型" dataKey="应用类型" stroke="#faad14" fill="#faad14" fillOpacity={0.5} />
            <Legend />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-container">
        <h3>指标柱状图</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="准确率" fill="#1890ff" />
            <Bar dataKey="精确率" fill="#52c41a" />
            <Bar dataKey="召回率" fill="#faad14" />
            <Bar dataKey="F1分数" fill="#f5222d" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MetricsChart;