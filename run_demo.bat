@echo off
echo Installing GIS Image Analysis Dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Dependencies installed successfully!
echo.
echo Running preprocessing demo...
python demo.py --preprocessing-only
echo.
echo Demo completed! Check the preprocessing_demo_results folder for outputs.
pause
