@echo off
REM Windows batch script to run the climate prediction project
REM Make sure conda is installed and in PATH

echo ========================================
echo Climate Prediction Project - Run Script
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not found in PATH
    echo Please install Anaconda or Miniconda and add it to PATH
    echo Or activate conda manually before running this script
    pause
    exit /b 1
)

REM Activate conda environment
echo [1/4] Activating conda environment...
call conda activate climate
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Environment 'climate' not found
    echo Creating environment...
    call conda env create -f environment.yml
    call conda activate climate
)

REM Check NOAA token
if "%NOAA_TOKEN%"=="" (
    echo [WARN] NOAA_TOKEN is not set. Set it via: setx NOAA_TOKEN "your_token_here"
) else (
    echo [INFO] NOAA token detected.
)

REM Prepare data
echo.
echo [2/4] Preparing data (Berkeley Earth + NOAA)...
python -m src.prepare_data --config configs/default.yaml --use-real-data
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data preparation failed
    pause
    exit /b 1
)

REM Train models
echo.
echo [3/4] Training models...
python -m src.train --config configs/default.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed
    pause
    exit /b 1
)

REM Evaluate models
echo.
echo [4/4] Evaluating models...
python -m src.evaluate --config configs/default.yaml
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Evaluation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo All steps completed successfully!
echo ========================================
echo Check outputs/figures/ for results
pause


