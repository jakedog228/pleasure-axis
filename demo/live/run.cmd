@echo off
rem Live steering demo: starts the local server on the GPU and opens http://127.0.0.1:8765
cd /d "%~dp0..\.."
python demo\live\server.py %*
