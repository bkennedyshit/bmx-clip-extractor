@echo off
setlocal
echo ============================================
echo  BMX Riding Extractor - Windows setup
echo ============================================

if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r workspace\requirements.txt

echo.
echo [done] Setup complete.
echo Run:
echo   venv\Scripts\python workspace\bmx_riding_extractor.py --list-models
echo.
endlocal
