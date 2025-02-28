@echo off

REM Define the main TeX file
set MAIN_TEX=main.tex

REM Check if the main.tex file exists
if not exist %MAIN_TEX% (
    echo Error: %MAIN_TEX% not found!
    exit /b 1
)

REM Compile the TeX file using pdflatex
pdflatex -interaction nonstopmode %MAIN_TEX%

REM Check if the compilation was successful
if %errorlevel% == 0 (
    echo Compilation successful!
) else (
    echo Compilation failed!
)

