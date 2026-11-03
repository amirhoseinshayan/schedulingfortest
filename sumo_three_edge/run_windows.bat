@echo off
cd /d "%~dp0"
python run_gui_links.py
if errorlevel 1 pause
