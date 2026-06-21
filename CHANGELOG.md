# Changelog

## 0.4.0
- **Canali MIDI 1–16 come Ableton**: l'interfaccia (schede di mapping e monitor
  MIDI) ora mostra i canali **1–16** invece di 0–15, per coincidere con la
  numerazione delle DAW. L'instradamento sul filo MIDI è invariato e corretto
  (verificato: il canale impostato è quello effettivamente trasmesso). Risolve
  la discrepanza per cui i canali in Ableton sembravano sfasati di 1.

## 0.3.0
- **Configurazioni complete salvabili**: dalla scheda MAPPING, "SAVE CONFIG" /
  "LOAD CONFIG" salvano e ricaricano l'intero setup di entrambi i dispositivi
  (profili, porte, loop e tutti i parametri di mapping) in
  `~/.sogno_cane/configs/`. (`sogno_cane/config.py`)
- **Icona dell'app**: logo onda EEG → nota, mostrato nella finestra/barra e
  usato dai collegamenti creati dall'installer. (`sogno_cane/assets/icon.ico`,
  generata da `tools/make_icon.py`)

## Non rilasciato (incluso in 0.3.0)
- **Aggiornamento automatico online** (`sogno_cane/update.py` +
  `portable/update_apply.py`): l'app controlla un manifest JSON, scarica la
  nuova versione con verifica SHA-256 e la applica al riavvio (con backup/
  rollback). Pulsante **UPDATES** per il controllo manuale; impostazioni
  `update_url` / `auto_update`. Strumento `tools/build_release.py` e guida
  `portable/AGGIORNAMENTI.txt`.
- **UI ridisegnata**: schede STUDIO / MAPPING / ARCHIVE, tema più compatto,
  forma d'onda ingrandita (prima schiacciata), niente più overlap.
- **Fix export CSV (ARCHIVE)**: il nome file ora è ripulito dai caratteri non
  validi su Windows (i due punti dell'orario); default in Downloads; `.csv`
  aggiunto in automatico.

## 0.2.0

### Funzionalità nuove
- **Registrazione & archivio**: pulsante `● REC` su ogni dispositivo; scheda
  `ARCHIVE` con rinomina, taglio (CUT di un intervallo), esportazione CSV,
  eliminazione e ricaricamento per riproduzione.
- **Caricamento file EEG** (`LOAD FILE`): riproduzione di `.edf`, `.bdf`,
  `.rec`, `.csv`, `.tsv`, `.txt`, `.npz`, `.npy` attraverso la pipeline MIDI.
  Lettore EDF/BDF pure-Python, validato bit-a-bit contro `pyedflib`.
- **Astrazione `PacketSource`**: il motore può essere alimentato dal simulatore
  o da un file (`ArrayPlaybackSource`).
- **Trasporto globale**: START ALL / STOP ALL / PANIC, aiuto in-app.
- **Persistenza impostazioni** (`settings.py`): profilo, porta MIDI, geometria
  finestra ricordati tra gli avvii.
- **Avvio robusto**: hook di eccezione con dialog + log; CLI `--selftest`,
  `--version`.

### Correzioni / miglioramenti
- **Pacing finalmente effettivo**: `min_interval_seconds`,
  `min_duration_seconds`, `change_threshold_norm` erano esposti nella UI ma
  ignorati dalle strategie (no-op). Ora il motore passa un orologio di
  simulazione deterministico (`WindowContext`) e le strategie lo rispettano.
  Densità di note ridotta da ~45/s a ~6/s.
- **Normalizzazione adattiva (AGC)** per le voci note e i CC: usa l'intera
  gamma indipendentemente dall'ampiezza assoluta del segnale.
- **Coerenza**: ritorna 0 con un solo segmento (prima sovrastimava a ~1).
- **Lockout in secondi** per clip launcher e threshold trigger (prima in
  "finestre", dipendente dall'hop).
- **Compatibilità all'indietro**: il mapper rileva strategie a 2 argomenti.
- Rimosso il codice duplicato/orfano `ui/` e `midi/` a livello di
  site-packages.
- Aggiunti `pyproject.toml`, `requirements.txt`, `README.md` e una suite di
  85 test (`pytest`).

## 0.1.0
- Versione iniziale: simulatore EEG human/dog, mappatura EEG→MIDI, UI PySide6.
