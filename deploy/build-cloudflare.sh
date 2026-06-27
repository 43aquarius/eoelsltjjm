#!/bin/bash
# ============================================
# Cloudflare Pages 静态导出部署脚本
# 使用方法: bash deploy/build-cloudflare.sh
# ============================================

set -e

echo "🚀 开始构建 Cloudflare Pages 静态导出..."

# 1. 安装依赖
echo "📦 安装依赖..."
npm install

# 2. 构建 + 导出静态文件
echo "🔨 构建并导出静态文件..."
npx next build

# 3. 导出为纯静态 HTML
echo "📄 导出静态文件到 dist/..."
npx next export -o dist

# 4. 添加 Cloudflare Pages 重定向规则
echo "📝 添加 _redirects 文件..."
cat > dist/_redirects << 'EOF'
/*    /index.html   200
EOF

# 5. 添加 _headers 缓存优化
echo "📝 添加 _headers 文件..."
cat > dist/_headers << 'EOF'
/*.js
  Cache-Control: public, max-age=31536000, immutable
/*.css
  Cache-Control: public, max-age=31536000, immutable
/*.png
  Cache-Control: public, max-age=31536000, immutable
/*.jpg
  Cache-Control: public, max-age=31536000, immutable
/*.svg
  Cache-Control: public, max-age=31536000, immutable
/*.ico
  Cache-Control: public, max-age=31536000, immutable
/*.woff2
  Cache-Control: public, max-age=31536000, immutable
/*.woff
  Cache-Control: public, max-age=31536000, immutable
EOF

echo ""
echo "✅ 构建完成！"
echo ""
echo "📂 输出目录: $(pwd)/dist"
echo ""
echo "🚀 部署方式（任选其一）："
echo ""
echo "  方式一：Cloudflare Dashboard 拖拽上传"
echo "    1. 打开 https://dash.cloudflare.com"
echo "    2. Workers & Pages → Create application → Pages → Upload assets"
echo "    3. 选择 dist 文件夹，点击 Deploy"
echo ""
echo "  方式二：Wrangler CLI"
echo "    npx wrangler pages deploy dist --project-name=local-contract"
echo ""
echo "  方式三：Git 集成自动部署"
echo "    1. 将代码推送到 GitHub"
echo "    2. Cloudflare Dashboard → Connect to Git"
echo "    3. 选择仓库，构建命令留空，输出目录填 dist"
echo ""
