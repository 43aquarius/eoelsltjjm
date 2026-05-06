import React, { useState } from 'react';
import AppLayout from './components/Layout/Layout';
import SinglePredict from './pages/SinglePredict/SinglePredict';
import BatchPredict from './pages/BatchPredict/BatchPredict';
import ModelEvaluation from './pages/ModelEvaluation/ModelEvaluation';
import History from './pages/History/History';
import './App.css';

const App: React.FC = () => {
  const [activeKey, setActiveKey] = useState('single-predict');

  const handleMenuChange = (key: string) => {
    setActiveKey(key);
  };

  const renderContent = () => {
    switch (activeKey) {
      case 'single-predict':
        return <SinglePredict />;
      case 'batch-predict':
        return <BatchPredict />;
      case 'model-evaluation':
        return <ModelEvaluation />;
      case 'history':
        return <History />;
      default:
        return <SinglePredict />;
    }
  };

  return (
    <AppLayout activeKey={activeKey} onMenuChange={handleMenuChange}>
      {renderContent()}
    </AppLayout>
  );
};

export default App;