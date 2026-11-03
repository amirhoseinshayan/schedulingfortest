@echo off
setlocal
set "ROOT=%~dp0.."
pushd "%ROOT%"

if "%1"=="" goto help
if /I "%1"=="validate" goto validate
if /I "%1"=="pilot" goto pilot
if /I "%1"=="train" goto train
if /I "%1"=="evaluate" goto evaluate
if /I "%1"=="tensorboard" goto tensorboard
if /I "%1"=="clean" goto clean

echo Unknown command: %1
goto help

:validate
python "without-gnn\validate.py"
if errorlevel 1 (popd & exit /b 1)
popd & exit /b 0

:pilot
python "without-gnn\train.py" --timesteps 20000
if errorlevel 1 (popd & exit /b 1)
popd & exit /b 0

:train
python "without-gnn\train.py" --timesteps 200000
if errorlevel 1 (popd & exit /b 1)
popd & exit /b 0

:evaluate
python "without-gnn\evaluate.py"
if errorlevel 1 (popd & exit /b 1)
popd & exit /b 0

:tensorboard
python -m tensorboard.main --logdir artifacts\without_gnn\logs\tensorboard
popd & exit /b 0

:clean
if exist artifacts\without_gnn rmdir /s /q artifacts\without_gnn
echo Removed artifacts\without_gnn.
popd & exit /b 0

:help
echo.
echo Usage: .\without-gnn\make.bat COMMAND
echo.
echo Commands:
echo   validate    Validate the shared env and prove policy ignores graph edges
 echo   pilot       Run a 20,000-step PPO without-GNN pilot
 echo   train       Run the full 200,000-step PPO without-GNN training
 echo   evaluate    Evaluate PPO_NO_GNN vs HEFT_GANG using legacy defaults
 echo   tensorboard Open without-GNN TensorBoard logs
 echo   clean       Remove ONLY artifacts\without_gnn
 echo.
popd & exit /b 0
