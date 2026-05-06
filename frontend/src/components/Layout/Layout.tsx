import React, { useState } from 'react';
import { Layout, Menu, Button, ConfigProvider } from 'antd';
import { MenuFoldOutlined, MenuUnfoldOutlined, HomeOutlined, FileTextOutlined, BarChartOutlined, HistoryOutlined } from '@ant-design/icons';
import './Layout.css';

const { Header, Sider, Content } = Layout;

interface LayoutProps {
  children: React.ReactNode;
  activeKey: string;
  onMenuChange: (key: string) => void;
}

const AppLayout: React.FC<LayoutProps> = ({ children, activeKey, onMenuChange }) => {
  const [collapsed, setCollapsed] = useState(false);

  const menuItems = [
    {
      key: 'single-predict',
      icon: <HomeOutlined />,
      label: '单文本预测',
    },
    {
      key: 'batch-predict',
      icon: <FileTextOutlined />,
      label: '批量预测',
    },
    {
      key: 'model-evaluation',
      icon: <BarChartOutlined />,
      label: '模型评估',
    },
    {
      key: 'history',
      icon: <HistoryOutlined />,
      label: '历史记录',
    },
  ];

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#1890ff',
          colorSuccess: '#52c41a',
          colorWarning: '#faad14',
          colorError: '#f5222d',
          colorBgLayout: '#f0f2f5',
          colorText: '#262626',
          colorTextSecondary: '#8c8c8c',
        },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header className="header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="logo">
            <h1>AI文本分析系统</h1>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Button
              type="text"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ marginRight: 16 }}
            />
          </div>
        </Header>
        <Layout>
          <Sider
            trigger={null}
            collapsible
            collapsed={collapsed}
            width={200}
            collapsedWidth={64}
            className="sidebar"
          >
            <Menu
              theme="dark"
              mode="inline"
              selectedKeys={[activeKey]}
              items={menuItems}
              onClick={(e) => onMenuChange(e.key)}
              style={{ height: '100%', borderRight: 0 }}
            />
          </Sider>
          <Layout style={{ padding: '24px' }}>
            <Content
              style={{
                padding: 24,
                margin: 0,
                minHeight: 280,
                background: '#fff',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.09)',
              }}
            >
              {children}
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
};

export default AppLayout;