@echo off
REM SOGNO_CANE - avvio portabile con auto-aggiornamento.
REM Doppio click. La finestra si apre dopo qualche secondo al primo avvio.
REM
REM Al primo avvio Windows SmartScreen potrebbe avvisare che il programma
REM non e' firmato: clicca "Ulteriori informazioni" -> "Esegui comunque".

cd /d "%~dp0"

REM 1) Applica un eventuale aggiornamento scaricato al precedente avvio.
runtime\python.exe "%~dp0update_apply.py"

REM 2) Avvia l'applicazione.
start "" "%~dp0runtime\pythonw.exe" -m sogno_cane
