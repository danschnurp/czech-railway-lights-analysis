@echo off

REM Script to install GT America font for fasthesis.cls on Windows
REM For use with TeXStudio and Chocolatey-installed MiKTeX

echo GT America Font Installation for LaTeX (fasthesis.cls)
echo ======================================================


REM Find the location of TEXMFLOCAL
set TEXMFLOCAL=%USERPROFILE%\texmf-local
if not exist "%TEXMFLOCAL%" (
    echo Could not determine TEXMFLOCAL location. Using default location...
    set TEXMFLOCAL=%USERPROFILE%\texmf-local
)

echo Using TEXMFLOCAL directory: %TEXMFLOCAL%

REM Create destination directories if they don't exist
mkdir "%TEXMFLOCAL%\fonts\truetype\gtamerica"
mkdir "%TEXMFLOCAL%\fonts\map\dvips\gtamerica"
mkdir "%TEXMFLOCAL%\fonts\enc\dvips\gtamerica"
mkdir "%TEXMFLOCAL%\fonts\tfm\gtamerica"
mkdir "%TEXMFLOCAL%\fonts\vf\gtamerica"
mkdir "%TEXMFLOCAL%\fonts\type1\gtamerica"
mkdir "%TEXMFLOCAL%\tex\latex\gtamerica"

REM Prompt user for the location of the fasthesis.zip file
set /p FASTHESIS_ZIP="Please enter the path to the fasthesis.zip file: "

if not exist "%FASTHESIS_ZIP%" (
    echo Error: File not found. Please verify the path.
    exit /b 1
)

REM Create a temporary directory for extraction
set TEMP_DIR=%TEMP%\gtamerica_temp
mkdir "%TEMP_DIR%"
echo Extracting to temporary directory: %TEMP_DIR%

REM Extract the zip file
powershell -Command "Expand-Archive -Path '%FASTHESIS_ZIP%' -DestinationPath '%TEMP_DIR%'"

REM Check if the install directory exists in the extracted contents
if not exist "%TEMP_DIR%\install\$TEXMFLOCAL" (
    echo Error: Expected directory structure not found in zip file.
    echo Looking for: %TEMP_DIR%\install\$TEXMFLOCAL
    rmdir /s /q "%TEMP_DIR%"
    exit /b 1
)

REM Copy the font files to the TEX installation
echo Copying font files to %TEXMFLOCAL%...
xcopy /E /I "%TEMP_DIR%\install\$TEXMFLOCAL\fonts" "%TEXMFLOCAL%\fonts\"
xcopy /E /I "%TEMP_DIR%\install\$TEXMFLOCAL\tex" "%TEXMFLOCAL%\tex\"

REM Add the map file to updmap.cfg
echo Updating font map configuration...
findstr /C:"Map GTAmerica" "%USERPROFILE%\AppData\Local\Programs\MiKTeX\miktex\config\updmap.cfg"
if %errorlevel% neq 0 (
    echo Map GTAmerica >> "%USERPROFILE%\AppData\Local\Programs\MiKTeX\miktex\config\updmap.cfg"
    echo Added 'Map GTAmerica' to updmap.cfg
) else (
    echo 'Map GTAmerica' already exists in updmap.cfg
)

REM Update the font maps
echo Regenerating font maps...
initexmf --mkmaps

REM Clean up
rmdir /s /q "%TEMP_DIR%"

echo.
echo GT America font has been installed successfully!
echo.
echo TeXStudio Configuration:
echo 1. Open TeXStudio
echo 2. Go to Options > Configure TeXStudio
echo 3. Select 'Commands' tab
echo 4. Make sure the paths point to your MiKTeX installation
echo    Typically at: C:\Program Files\MiKTeX\miktex\bin\x64
echo 5. Click OK and restart TeXStudio
echo.
echo You should now be able to use the fasthesis.cls with GT America font.

pause
