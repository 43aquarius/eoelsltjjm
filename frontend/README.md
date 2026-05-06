# AI文本分析系统前端

基于React 18+、TypeScript和Ant Design 5.x开发的AI文本分析前端应用，用于分析文本中是否存在AI应用、AI使用方式和AI应用类型。

## 技术栈

- **前端框架**：React 18+ 与 TypeScript
- **UI组件库**：Ant Design 5.x
- **状态管理**：React Hooks (useState, useEffect)
- **HTTP客户端**：Axios
- **构建工具**：Vite
- **数据可视化**：ECharts, Recharts
- **性能优化**：react-window

## 功能模块

1. **单文本预测**：输入文本，选择模型，查看预测结果
2. **批量预测**：上传Excel/CSV文件，批量分析文本
3. **模型评估**：评估模型性能，查看详细指标和可视化图表
4. **历史记录**：查看和管理历史预测记录

## 快速开始

### 安装依赖

```bash
npm install
```

### 开发模式运行

```bash
npm run dev
```

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 项目结构

```
frontend/
├── src/
│   ├── components/          # 可复用组件
│   │   ├── Layout/         # 布局组件
│   │   ├── PredictCard/    # 预测结果卡片
│   │   ├── FileUpload/     # 文件上传
│   │   └── Charts/         # 图表组件
│   ├── pages/              # 页面组件
│   │   ├── SinglePredict/  # 单文本预测
│   │   ├── BatchPredict/   # 批量预测
│   │   ├── ModelEvaluation/ # 模型评估
│   │   └── History/        # 历史记录
│   ├── services/           # API服务
│   │   └── api.ts          # API接口定义
│   ├── types/              # TypeScript类型定义
│   ├── utils/              # 工具函数
│   ├── hooks/              # 自定义Hooks
│   ├── styles/             # 全局样式
│   ├── App.tsx             # 应用主组件
│   └── main.tsx            # 应用入口
├── public/                 # 静态资源
├── package.json            # 项目配置
├── tsconfig.json           # TypeScript配置
├── vite.config.ts          # Vite配置
└── README.md               # 项目说明
```

## API接口

前端应用需要与后端API进行交互，API接口规范如下：

### 基础URL

```
http://localhost:8000/api
```

### 接口列表

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | /predict | 单文本预测 |
| POST | /batch-predict | 批量预测 |
| POST | /evaluate | 模型评估 |
| GET | /models | 获取可用模型列表 |
| GET | /history | 获取历史记录 |
| DELETE | /history/:id | 删除历史记录 |
| POST | /upload | 文件上传 |
| GET | /download/:filename | 结果下载 |

## 响应式设计

- **桌面端** (≥1200px)：完整布局，侧边栏固定显示
- **平板端** (768px-1199px)：自适应布局，侧边栏可折叠
- **移动端** (<768px)：抽屉式导航，卡片式布局

## 性能优化

- 使用React.memo优化组件渲染
- 大数据列表使用虚拟滚动
- 图片和资源懒加载
- API请求防抖处理
- 骨架屏加载状态

## 浏览器支持

- Chrome (最新版本)
- Firefox (最新版本)
- Safari (最新版本)
- Edge (最新版本)
