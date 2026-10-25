@echo off
title 桑叶病虫害检测系统
cd /d "%~dp0"

set PY=%~dp0.venv\Scripts\python.exe
if not exist "%PY%" set PY=python

echo ============================================================
echo   桑叶病虫害检测系统
echo.
echo   正在启动, 稍后浏览器会自动打开 http://127.0.0.1:7860
echo   使用完毕后关闭本窗口即可停止服务
echo ============================================================
echo.

"%PY%" app.py

echo.
echo 服务已退出。若上方有报错信息, 请截图反馈。
pause
