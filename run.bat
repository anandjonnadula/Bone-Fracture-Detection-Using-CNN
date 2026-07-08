@echo off
REM Launch the Bone Fracture Detection web app.
REM The virtualenv lives at %USERPROFILE%\.venvs\bfd (a SHORT path) because
REM installing TensorFlow into a venv nested inside this deeply-nested project
REM folder exceeds the Windows 260-character path limit.

set PYTHON=%USERPROFILE%\.venvs\bfd\Scripts\python.exe

if not exist "%PYTHON%" (
    echo Virtualenv not found at %USERPROFILE%\.venvs\bfd
    echo Create it first:
    echo     python -m venv %%USERPROFILE%%\.venvs\bfd
    echo     %%USERPROFILE%%\.venvs\bfd\Scripts\pip install -r requirements.txt
    exit /b 1
)

cd /d "%~dp0"
"%PYTHON%" app.py
