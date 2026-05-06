@echo off
echo Starting AI Text Analysis Server...
echo.

REM 激活虚拟环境
call D:\other\tool\anaconda\envs\pytorch\Scripts\activate.bat

REM 启动后端服务器
echo Starting backend server on http://localhost:8000
python app.py

pause
