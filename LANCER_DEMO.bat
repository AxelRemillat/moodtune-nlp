@echo off
chcp 65001 >nul
title MoodTune NLP - Demo
cd /d "%~dp0"

REM 1) Trouver Python (Anaconda en priorite, sinon python du PATH)
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"

REM 2) Environnement virtuel du projet (cree une seule fois)
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creation de l'environnement Python du projet...
    "%PY%" -m venv .venv
)
set "VPY=.venv\Scripts\python.exe"

REM 3) Dependances (installees une seule fois, quelques minutes)
"%VPY%" -c "import streamlit, sentence_transformers, plotly, sklearn" 2>nul
if errorlevel 1 (
    echo [2/3] Installation des dependances, patiente quelques minutes...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    "%VPY%" -m pip install -r requirements.txt
)

REM 4) Lancement (le modele SBERT se telecharge au 1er lancement puis reste en cache)
echo [3/3] Lancement de MoodTune NLP sur http://localhost:8501
"%VPY%" -m streamlit run app.py --server.headless false
pause
