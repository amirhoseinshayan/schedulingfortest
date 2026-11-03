@echo off
cd /d "%~dp0"
set MUX_MOBILITY3=1
python -m muxgnn.mobility3.validate_static || goto fail
python -m muxgnn.mobility3.capture --seconds 600 || goto fail
python -m muxgnn.mobility3.generate --scenarios 24 || goto fail
python -m muxgnn.mobility3.heft_online || goto fail
echo Dataset build and baseline analysis completed.
pause
exit /b 0
:fail
echo ERROR: review the output above. Nothing was falsely marked as valid.
pause
exit /b 1
