@echo off
echo ============================================
echo 复制静态文件到桌面部署文件夹
echo ============================================

set SRC=D:\file\project\LocalContract\local-contract\out
set DST=C:\Users\23529\Desktop\local-contract-cloudflare

if not exist "%DST%" mkdir "%DST%"

echo 复制文件...
xcopy "%SRC%\*" "%DST%\" /E /I /Y /Q

echo.
echo 完成！文件已复制到: %DST%
echo.
dir "%DST%" /B
echo.
echo ============================================
echo 部署说明:
echo 1. 打开 https://dash.cloudflare.com
echo 2. Workers & Pages -> Create application -> Pages -> Upload assets
echo 3. 选择文件夹: %DST%
echo 4. 点击 Deploy site
echo ============================================
pause
