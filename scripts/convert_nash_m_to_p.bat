@echo off
setlocal
if not defined BEVFUSION_WSL_DISTRO set "BEVFUSION_WSL_DISTRO=Ubuntu-26.04"
pushd "%~dp0.."
wsl.exe --distribution %BEVFUSION_WSL_DISTRO% --exec bash scripts/convert_nash_m_to_p.sh %*
set "TASK_EXIT=%errorlevel%"
popd
exit /b %TASK_EXIT%
