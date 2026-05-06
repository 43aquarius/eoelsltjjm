import React, { useState } from 'react';
import { Upload, Button, message, Progress, Tag } from 'antd';
import { InboxOutlined, FileExcelOutlined, CheckCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import './FileUpload.css';

const { Dragger } = Upload;

interface FileUploadProps {
  accept: string;
  maxSize: number;
  onFileUpload: (file: File) => void;
  disabled?: boolean;
  requiredColumns?: string[];
  description?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  accept,
  maxSize,
  onFileUpload,
  disabled = false,
  requiredColumns = [],
  description = ''
}) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleUpload = async (file: File) => {
    if (file.size > maxSize) {
      message.error(`文件大小不能超过 ${maxSize / 1024 / 1024}MB`);
      return false;
    }

    setUploading(true);
    setProgress(0);

    const totalSteps = 20;
    let currentStep = 0;

    const interval = setInterval(() => {
      currentStep++;
      const newProgress = Math.min((currentStep / totalSteps) * 100, 100);
      setProgress(newProgress);

      if (currentStep >= totalSteps) {
        clearInterval(interval);
        setTimeout(() => {
          setUploadedFile(file);
          onFileUpload(file);
          setUploading(false);
          message.success(`文件 ${file.name} 已准备就绪`);
        }, 100);
      }
    }, 50);

    return false;
  };

  const handleRemove = () => {
    setUploadedFile(null);
    setProgress(0);
    message.info('已移除文件');
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <div className="file-upload-wrapper">
      <Dragger
        accept={accept}
        showUploadList={false}
        beforeUpload={handleUpload}
        disabled={disabled || uploading || !!uploadedFile}
      >
        {uploading ? (
          <div className="upload-progress">
            <Progress
              percent={Math.round(progress)}
              status={progress === 100 ? "success" : "active"}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
            <p style={{ marginTop: 16, textAlign: 'center', color: '#666' }}>
              正在准备文件...
            </p>
          </div>
        ) : uploadedFile ? (
          <div className="upload-success">
            <p className="ant-upload-drag-icon" style={{ color: '#52c41a' }}>
              <CheckCircleOutlined />
            </p>
            <p className="ant-upload-text" style={{ color: '#52c41a', fontWeight: 500 }}>
              文件已准备就绪
            </p>
            <div style={{ marginTop: 16, padding: '12px 24px', background: '#f6ffed', borderRadius: '4px', border: '1px solid #b7eb8f' }}>
              <p style={{ margin: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                <FileExcelOutlined style={{ fontSize: '18px', color: '#52c41a' }} />
                <span style={{ fontWeight: 500 }}>{uploadedFile.name}</span>
                <Tag color="green">{formatFileSize(uploadedFile.size)}</Tag>
              </p>
            </div>
            <Button
              type="link"
              danger
              icon={<DeleteOutlined />}
              onClick={(e) => {
                e.stopPropagation();
                handleRemove();
              }}
              style={{ marginTop: 12 }}
            >
              移除文件
            </Button>
          </div>
        ) : (
          <div className="upload-content">
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
            <p className="ant-upload-hint">
              支持 {accept} 格式，单个文件不超过 {maxSize / 1024 / 1024}MB
            </p>
          </div>
        )}
      </Dragger>

      {(requiredColumns.length > 0 || description) && (
        <div style={{
          marginTop: 16,
          padding: '12px 16px',
          background: '#f0f5ff',
          borderRadius: '4px',
          border: '1px solid #adc6ff'
        }}>
          <div style={{ fontSize: '14px', fontWeight: 500, color: '#1890ff', marginBottom: 8 }}>
            📋 文件格式要求
          </div>
          {description && (
            <div style={{ fontSize: '13px', color: '#595959', marginBottom: 8 }}>
              {description}
            </div>
          )}
          {requiredColumns.length > 0 && (
            <div style={{ fontSize: '13px', color: '#595959' }}>
              <span style={{ fontWeight: 500 }}>必需列名：</span>
              {requiredColumns.map((col, index) => (
                <Tag key={index} color="blue" style={{ marginLeft: 4, marginTop: 4 }}>
                  {col}
                </Tag>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FileUpload;