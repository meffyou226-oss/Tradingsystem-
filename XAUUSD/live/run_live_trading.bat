@echo off
setlocal enabledelayedexpansion
REM ====================================================================
REM XAUUSD XGBoost M1 Live-Trading Bot - Setup & Run
REM ====================================================================
REM Diese Batch-Datei:
REM 1. Findet Python und pip
REM 2. Installiert alle benötigten Bibliotheken
REM 3. Trainiert das XGBoost-Modell (falls nicht vorhanden)
REM 4. Startet den Live-Trading-Bot mit MT5-Anbindung
REM ===

set SCRIPT_DIR=%~dp0
set XAUUSD_DIR=%SCRIPT_DIR%..
set PYTHONPATH=%XAUUSD_DIR%;%SCRIPT_DIR%

echo.
echo ============================================================
echo  XGBoost XAUUSD M1 Live-Trading Bot
echo  Setup & Run Script
echo ============================================================
echo.

REM === Pruefe Python ===
echo [1/4] Suche Python...
python --version 2>nul
if errorlevel 1 (
    echo FEHLER: Python nicht gefunden!
    echo Loesung: Installieren Sie Python von https://python.org
    echo Wichtig: Waehlen Sie bei der Installation "Add Python to PATH"!
    echo.
    pause
    exit /b 1
)
echo [OK] Python gefunden

echo.
echo [2/4] Installiere Python-Bibliotheken...
echo ------------------------------------------------
echo Verwende: python -m pip

echo   Upgrade pip...
python -m pip install --upgrade pip 2>nul

echo   Installiere: numpy (Version ^<2) ...
python -m pip install "numpy<2"
if errorlevel 1 (
    echo FEHLER bei numpy Installation
    pause
    exit /b 1
)

echo   Installiere: pandas, scikit-learn...
python -m pip install pandas scikit-learn
if errorlevel 1 (
    echo FEHLER bei pandas/scikit-learn Installation
    pause
    exit /b 1
)

echo   Installiere: xgboost...
python -m pip install xgboost
if errorlevel 1 (
    echo FEHLER bei xgboost Installation
    pause
    exit /b 1
)

echo   Installiere: MetaTrader5 Python API...
python -m pip install MetaTrader5
if errorlevel 1 (
    echo FEHLER bei MetaTrader5 Installation
    pause
    exit /b 1
)

echo   Installiere: joblib, scipy...
python -m pip install joblib scipy
echo.

echo [OK] Alle Bibliotheken installiert
echo.

REM === Model-Training ===
echo [3/4] Modell-Training...
echo ------------------------------------------------
if exist "%SCRIPT_DIR%xgboost_model.pkl" (
    echo   Modell bereits vorhanden: xgboost_model.pkl
) else (
    echo   Modell nicht gefunden. Trainiere XGBoost...
    echo   Wechseln zu: %XAUUSD_DIR%
    cd /d "%XAUUSD_DIR%"
    echo   Starte Training: src/05_xgboost_train.py
    python src/05_xgboost_train.py
    if errorlevel 1 (
        echo FEHLER beim Training!
        echo Moegliche Ursachen:
        echo   - Trainingsdaten nicht vorhanden
        echo   - Fehlende Bibliotheken
        echo.
        echo Manuelle Loesung: Das Modell wurde bereits einmal trainiert
        echo und ist im Ordner models/ gespeichert. Kopieren Sie models\xgboost.pkl
        echo nach live\xgboost_model.pkl
        echo.
        cd /d "%SCRIPT_DIR%"
        echo.
        pause
        exit /b 1
    )
    copy "%XAUUSD_DIR%\models\xgboost.pkl" "%SCRIPT_DIR%xgboost_model.pkl" >nul
    echo   Modell trainiert und kopiert.
    cd /d "%SCRIPT_DIR%"
)
echo.

REM === Live-Trading ===
echo [4/4] Starte Live-Trading...
echo ------------------------------------------------
echo   WICHTIG: MetaTrader 5 muss bereits geoeffnet und mit
echo   Kontoverbindung sein!
echo.
echo   Oeffnen Sie MT5 nun und bestaetigen Sie mit beliebiger Taste...
pause >nul

cd /d "%SCRIPT_DIR%"
python live_trader.py
set EXIT_CODE=%errorlevel%

echo.
echo ============================================================
echo  Trading Bot wurde beendet (Exit-Code=%EXIT_CODE%)
echo  Log-Datei: %SCRIPT_DIR%live_trading.log
echo ============================================================

if %EXIT_CODE% NEQ 0 (
    echo FEHLER beim Trading-Bot!
    echo Pruefen Sie die Log-Datei fuer Details.
)

echo.
pause
endlocal
