@echo off
echo Building PicGo Local...
echo.

REM Activate venv if not already active (optional check)
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist PicGo_Local.spec del PicGo_Local.spec

REM Run PyInstaller
REM --noconsole: Don't show terminal window (GUI app)
REM --onefile: Bundle everything into one .exe
REM --name: Name of the output .exe
REM --hidden-import: Ensure these modules are included
pyinstaller --noconsole --onefile --name "PicGo_Local" ^
    --hidden-import=torch ^
    --hidden-import=diffusers ^
    --hidden-import=transformers ^
    --hidden-import=accelerate ^
    --hidden-import=PIL ^
    --hidden-import=tkinter ^
    picgo/picgo_local.py

echo.
echo Build complete!
echo Executable is in the 'dist' folder.
echo.
pause
