@echo off
cd /d "%~dp0"
where sumo-gui >nul 2>&1
if errorlevel 1 (
 echo Could not locate sumo-gui. Set SUMO_HOME and add SUMO\bin to PATH.
 pause
 exit /b 1
)
sumo-gui -c six_vehicles_three_servers.sumocfg
