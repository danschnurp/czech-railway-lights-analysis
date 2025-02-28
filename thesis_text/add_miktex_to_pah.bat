@echo off

REM Script to add MiKTeX to the system PATH environment variable

REM Define the MiKTeX binary directory
REM Adjust the path if MiKTeX is installed in a different location
set MIKTEX_PATH=C:\Program Files\MiKTeX\miktex\bin\x64

REM Check if MiKTeX is installed at the specified path
if not exist "%MIKTEX_PATH%\pdflatex.exe" (
    echo Error: MiKTeX not found at %MIKTEX_PATH%.
    echo Please verify the installation path or install MiKTeX.
    exit /b 1
)

REM Add MiKTeX to the system PATH
echo Adding MiKTeX to the system PATH...
setx /M PATH "%PATH%;%MIKTEX_PATH%"

REM Verify the PATH update
echo Verifying PATH update...
where pdflatex
if %errorlevel% == 0 (
    echo MiKTeX has been successfully added to the system PATH.
) else (
    echo Failed to add MiKTeX to the system PATH.
)

REM Note for the user
echo Please open a new Command Prompt window to use the updated PATH.

pause
