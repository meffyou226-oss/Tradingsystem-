@echo off
REM ============================================================
REM XAUUSD Live Trading Bot - Installer und Starter
REM ============================================================
REM Dieses Skript installiert alle benötigten Bibliotheken
REM und startet den Trading Bot.
REM ============================================================

echo.
echo ============================================
echo  XAUUSD LIVE TRADING BOT - Setup
echo ============================================
echo.

REM Prüfe ob Python installiert ist
python --version >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python ist nicht installiert!
    echo Bitte installiere Python 3.10 oder neuer von:
    echo https://www.python.org/downloads/
    echo.
    echo WICHTI


G: Aktiviere "Add Python to PATH" bei der Installation!
    pause
    exit /b 1
)

echo [1/4] Python gefunden:
python --version
echo.

REM Virtuelle Umgebung erstellen (optional)
echo [2/4] Erstelle virtuelle Umgebung...
if not exist venv (
    python -m venv venv
    echo   Virtuelle Umgebung erstellt.
) else (
    echo   Virtuelle Umgebung existiert bereits.
)
echo.

REM Umgebung aktivieren
call venv\Scripts\activate

REM Pip upgraden
echo [3/4] Upgrade pip...
python -m pip install --upgrade pip
echo.

REM Bibliotheken installieren
echo [4/4] Installiere benoetigte Bibliotheken...
pip install -r requirements.txt
if errorlevel 1 (
    echo [FEHLER] Installation fehlgeschlagen!
    pause
    exit /b 1
)
echo.
echo   Alle Bibliotheken installiert!
echo.

REM ============================================================
REM Bot starten
REM ============================================================
echo ============================================
echo  Starte Trading Bot...
echo ============================================
echo.
echo  WICHTIG:
echo  - MetaTrader 5 muss geoeffnet und angemeldet sein
echo  - Das Symbol XAUUSD muss im Market Watch sein
echo  - Druecke Ctrl+C zum Stoppen
echo.
echo ============================================

python src/live_bot.py

echo.
echo Bot beendet.
pause
