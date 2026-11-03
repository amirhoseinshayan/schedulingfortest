@echo off
setlocal

if "%1"=="" goto help

if /I "%1"=="generate" goto generate
if /I "%1"=="validate" goto validate
if /I "%1"=="pilot" goto pilot
if /I "%1"=="train" goto train
if /I "%1"=="evaluate" goto evaluate
if /I "%1"=="tensorboard" goto tensorboard
if /I "%1"=="clean" goto clean

echo Unknown command: %1
goto help

:generate
python run_legacy_gat.py generate
if errorlevel 1 exit /b 1
exit /b 0

:validate
python run_legacy_gat.py validate
if errorlevel 1 exit /b 1
exit /b 0

:pilot
python run_legacy_gat.py train --timesteps 20000
if errorlevel 1 exit /b 1
exit /b 0

:train
python run_legacy_gat.py train --timesteps 200000
if errorlevel 1 exit /b 1
exit /b 0

:evaluate
python run_legacy_gat.py evaluate
if errorlevel 1 exit /b 1
exit /b 0

:tensorboard
python -m tensorboard.main --logdir artifacts\legacy_gat\logs\tensorboard
exit /b 0

:clean
if exist artifacts\legacy_gat rmdir /s /q artifacts\legacy_gat
echo Removed artifacts\legacy_gat.
exit /b 0

:help
echo.
echo Usage: .\make.bat COMMAND
echo.
echo Commands:
echo   generate    Generate the legacy dataset from data\legacy\raw_results
echo   validate    Run check_env + HEFT_GANG scheduling invariants
echo   pilot       Run a 20,000-step PPO+GAT+Gang pilot
echo   train       Run the full 200,000-step training
echo   evaluate    Evaluate PPO_GANG vs HEFT_GANG
echo   tensorboard Open TensorBoard logs
echo   clean       Remove ONLY artifacts\legacy_gat
echo.
exit /b 0
