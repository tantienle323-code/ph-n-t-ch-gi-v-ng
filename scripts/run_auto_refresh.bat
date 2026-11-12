@echo off
setlocal
cd /d "%~dp0.."
python scripts\auto_refresh.py --interval-minutes 1440