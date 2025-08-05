@echo off
echo Setting up GIS Image Analysis Environment...
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv gis_env

REM Activate virtual environment
echo Activating virtual environment...
call gis_env\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
python -m pip install -r requirements.txt

echo.
echo Setup completed successfully!
echo.
echo To activate the environment, run: gis_env\Scripts\activate.bat
echo To run the demo, use: python demo.py --preprocessing-only
echo.
pause
