@echo off
cd /d "%~dp0"
set MUX_MOBILITY3=1
python -m muxgnn.mobility3.pretrain --epochs 10 --batch-size 4 || goto fail
python -m muxgnn.mobility3.train --timesteps 20000 || goto fail
echo Training completed.
pause
exit /b 0
:fail
pause
exit /b 1
