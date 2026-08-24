@echo off
REM ====================================================================
REM XAUUSD XGBoost M1 Live-Trading Bot - Setup & Run
REM ====================================================================
REM Diese Batch-Datei:
REM 1. Installiert alle benötigten Python-Bibliotheken
REM 2. Trainiert das XGBoost-Modell (falls nicht vorhanden)
REM 3. Startet den Live-Trading-Bot mit MT5-Anbindung
REM ====================================================================

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set XAUUSD_DIR=%SCRIPT_DIR%..
set PYTHONPATH=%SCRIPT_DIR%..;%SCRIPT_DIR%

echo.
echo ============================================================
echo  XGBoost XAUUSD M1 Live-Trading Bot
echo  Setup & Run Script
echo ============================================================
echo.

REM === Pruefe Python ===
python --version >nul 2>&1
if errorlevel 1 (
    echo FEHLER: Python nicht gefunden. Bitte installieren von python.org
    pause
    exit /b 1
)
echo [OK] Python gefunden

REM === Installiere Bibliotheken ===
echo.
echo [1/3] Installiere Python-Bibliotheken...
echo ------------------------------------------------

echo   Python-Pfad wird gesucht...
set PIP_CMD=python -m pip

!PIP_CMD! install --quiet --upgrade pip

echo   Installiere: numpy, pandas, scikit-learn...
!PIP_CMD! install --quiet "numpy<2"
!PIP_CMD! install --quiet pandas scikit-learn
if errorlevel 1 (
    echo FEHLER bei Core-Bibliotheken
    pause
    exit /b 1
)

echo   Installiere: xgboost...
!PIP_CMD! install --quiet xgboost
if errorlevel 1 (
    echo FEHLER: xgboost Installation fehlgeschlagen
    pause
    exit /b 1
)

echo   Installiere: MetaTrader5...
!PIP_CMD! install --quiet MetaTrader5
if errorlevel 1 (
    echo FEHLER: MetaTrader5 Installation fehlgeschlagen
    echo Stellen Sie sicher, dass MetaTrader 5 installiert ist.
    pause
    exit /b 1
)

echo   Installiere: joblib, scipy...
!PIP_CMD! install --quiet joblib scipy
if errorlevel 1 (
    echo WARNUNG: joblib/scipy Installation fehlgeschlagen
)

echo.
echo [OK] Alle Bibliotheken installiert
echo.

REM === Trainiere Modell (falls nicht vorhanden) ===
echo [2/3] Modell-Training...
echo ------------------------------------------------
if exist "%SCRIPT_DIR%xgboost_model.pkl" (
    echo   Modell bereits vorhanden: xgboost_model.pkl
) else (
    echo   Trainiere XGBoost-Modell...
    set PYTHONPATH=%XAUUSD_DIR%;%SCRIPT_DIR%
    cd /d "%XAUUSD_DIR%"
    python "%SCRIPT_DIR%..\src\05_xgboost_train.py"
    if errorlevel 1 (
        echo FEHLER beim Training
        cd /d "%SCRIPT_DIR%"
        pause
        exit /b 1
    )
    copy "%XAUUSD_DIR%\models\xgboost.pkl" "%SCRIPT_DIR%xgboost_model.pkl" >nul
    echo   Modell trainiert und kopiert.
    cd /d "%SCRIPT_DIR%"
)
echo.

REM === Pruefe MetaTrader 5 ===
echo [3/3] Starte Live-Trading...
echo ------------------------------------------------
echo   WICHTIG: MetaTrader 5 muss geöffnet sein!
echo   Das Terminal muss mit Ihrem Konto verbunden sein.
echo.
echo   Oeffnen Sie MT5 und bestaetigen Sie mit beliebiger Taste...
pause
echo.

python "%SCRIPT_DIR%live_trader.py"
if errorlevel 1 (
    echo.
    echo FEHLER beim Trading-Bot
    echo Protokolldatei: %SCRIPT_DIR%live_trading.log
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Trading Bot wurde gestoppt.
echo  Log-Datei: %SCRIPT_DIR%live_trading.log
echo ============================================================
pause
