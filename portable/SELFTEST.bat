@echo off
REM SOGNO_CANE - diagnostica. Esegue la pipeline senza aprire la finestra
REM e stampa eventuali errori. Manda l'output a chi ti aiuta col debug.

cd /d "%~dp0"
echo === SOGNO_CANE selftest ===
runtime\python.exe -m sogno_cane --selftest
echo.
echo === Porte MIDI viste da rtmidi ===
runtime\python.exe -c "import rtmidi; o=rtmidi.MidiOut(); print([o.get_port_name(i) for i in range(o.get_port_count())])"
echo.
echo === FINE - premi un tasto per chiudere ===
pause >nul
