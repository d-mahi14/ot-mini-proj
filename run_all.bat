@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM run_all.bat: Fair and Load-Balanced Student-Alumni Matching at VJTI
REM Windows Automation Runner for All Pipeline Targets
REM ==============================================================================

set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

echo ==============================================================================
echo       VJTI STUDENT-ALUMNI MULTI-OBJECTIVE MATCHING OPTIMIZATION
echo       Capacitated Assignment with Rawlsian Fairness ^& Branch Parity
echo       Department of Production ^& Industrial Engineering / Computer Eng.
echo       Veermata Jijabai Technological Institute (VJTI), Mumbai
echo ==============================================================================
echo.

REM Verify Python is installed and accessible
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.11+ and add it to PATH.
    exit /b 1
)

REM Route to requested target
if /i "%TARGET%"=="help" goto target_help
if /i "%TARGET%"=="install" goto target_install
if /i "%TARGET%"=="test" goto target_test
if /i "%TARGET%"=="data" goto target_data
if /i "%TARGET%"=="baselines" goto target_baselines
if /i "%TARGET%"=="exact" goto target_exact
if /i "%TARGET%"=="benchmark" goto target_benchmark
if /i "%TARGET%"=="sensitivity" goto target_sensitivity
if /i "%TARGET%"=="interpret" goto target_interpret
if /i "%TARGET%"=="figures" goto target_figures
if /i "%TARGET%"=="clean" goto target_clean
if /i "%TARGET%"=="all" goto target_all

echo [ERROR] Unknown target: %TARGET%
goto target_help

REM ------------------------------------------------------------------------------
:target_help
echo Usage: run_all.bat [target]
echo.
echo Available Targets:
echo   install      - Install project dependencies from requirements.txt
echo   test         - Run pytest verification suite on tests/
echo   data         - Generate / verify synthetic benchmark datasets in data/
echo   baselines    - Run baseline heuristics (Random, Greedy, Min-Cost Flow)
echo   exact        - Run exact HiGHS MILP Pareto frontier and secondary scenario
echo   benchmark    - Run multi-objective evolutionary algorithms benchmark
echo   sensitivity  - Run parameter sensitivity and coverage-gap analysis
echo   interpret    - Run Pareto frontier interpretation and institutional policy report
echo   figures      - Generate visualizations and compile publication-grade PDFs
echo   all          - Execute full end-to-end research and verification pipeline
echo   clean        - Remove temporary build files, bytecode, and test caches
exit /b 0

REM ------------------------------------------------------------------------------
:target_install
echo [1/1] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Dependencies installed successfully.
exit /b 0

REM ------------------------------------------------------------------------------
:target_test
echo [1/1] Running pytest verification suite...
python -m pytest tests/ -v
if errorlevel 1 (
    echo [ERROR] Test suite encountered failures.
    exit /b %errorlevel%
)
echo [SUCCESS] All unit tests passed!
exit /b 0

REM ------------------------------------------------------------------------------
:target_data
echo [1/1] Generating synthetic datasets in data/...
python -m src.data_generator
if errorlevel 1 (
    echo [ERROR] Data generation failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Datasets generated/verified.
exit /b 0

REM ------------------------------------------------------------------------------
:target_baselines
echo [1/1] Running baseline algorithms...
python main.py --mode baselines
if errorlevel 1 (
    echo [ERROR] Baseline execution failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Baselines executed successfully.
exit /b 0

REM ------------------------------------------------------------------------------
:target_exact
echo [1/1] Running exact HiGHS MILP analysis...
python main.py --mode exact
if errorlevel 1 (
    echo [ERROR] Exact MILP execution failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Exact MILP completed.
exit /b 0

REM ------------------------------------------------------------------------------
:target_benchmark
echo [1/1] Running multi-objective evolutionary benchmark...
python main.py --mode run
if errorlevel 1 (
    echo [ERROR] Benchmark execution failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Benchmark completed.
exit /b 0

REM ------------------------------------------------------------------------------
:target_sensitivity
echo [1/1] Running parameter sensitivity and coverage gap analysis...
python main.py --mode sensitivity
if errorlevel 1 (
    echo [ERROR] Sensitivity analysis failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Sensitivity analysis completed.
exit /b 0

REM ------------------------------------------------------------------------------
:target_interpret
echo [1/1] Running Pareto interpretation pipeline...
python main.py --mode interpretation
if errorlevel 1 (
    echo [ERROR] Interpretation pipeline failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Interpretation pipeline completed.
exit /b 0

REM ------------------------------------------------------------------------------
:target_figures
echo [1/1] Generating figures and PDF deliverables...
python generate_figures.py
if errorlevel 1 (
    echo [ERROR] Figure / PDF generation failed.
    exit /b %errorlevel%
)
echo [SUCCESS] Figures and PDFs compiled.
exit /b 0

REM ------------------------------------------------------------------------------
:target_clean
echo [1/1] Cleaning temporary files and caches...
python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; [p.unlink() for p in pathlib.Path('.').rglob('.coverage*')]; print('Build artifacts and caches cleaned.')"
echo [SUCCESS] Workspace clean.
exit /b 0

REM ------------------------------------------------------------------------------
:target_all
echo >>> STARTING FULL RESEARCH PIPELINE AND VERIFICATION <<<
echo.

echo --- Step 1: Synthetic Dataset Verification ---
python -m src.data_generator
if errorlevel 1 (
    echo [ERROR] Step 1 Data verification failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 2: Running Unit Tests ---
python -m pytest tests/ -v
if errorlevel 1 (
    echo [ERROR] Step 2 Unit tests failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 3: Baseline Algorithms Benchmark ---
python main.py --mode baselines
if errorlevel 1 (
    echo [ERROR] Step 3 Baselines failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 4: Exact HiGHS MILP Pareto Analysis ---
python main.py --mode exact
if errorlevel 1 (
    echo [ERROR] Step 4 Exact MILP failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 5: Multi-Objective Evolutionary Algorithms Benchmark ---
python main.py --mode run --config experiments/configs/small_50x30.yaml
if errorlevel 1 (
    echo [ERROR] Step 5 Benchmark failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 6: Parameter Sensitivity ^& Coverage Gaps ---
python main.py --mode sensitivity
if errorlevel 1 (
    echo [ERROR] Step 6 Sensitivity analysis failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 7: Pareto Interpretation ^& Institutional Policy ---
python main.py --mode interpretation --instance small_50x30
if errorlevel 1 (
    echo [ERROR] Step 7 Interpretation failed.
    exit /b %errorlevel%
)

echo.
echo --- Step 8: PDF Report ^& Visual Deliverables Generation ---
python generate_figures.py
if errorlevel 1 (
    echo [WARNING] Headless PDF generation encountered an issue; visual figures remain available in figures/.
)

echo.
echo ==============================================================================
echo    FULL RESEARCH PIPELINE COMPLETED SUCCESSFULLY ON WINDOWS!
echo ==============================================================================
exit /b 0

