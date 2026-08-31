# SOGNO_CANE

**EEG → MIDI** per uomo e cane (g.tec Unicorn Hybrid Black), con simulatore
integrato, registrazione/archivio dei segnali e riproduzione di file EEG.

SOGNO_CANE genera (o legge) flussi EEG a 8 canali, ne estrae la potenza nelle
bande cerebrali (delta / theta / alpha / beta / gamma) e le traduce in musica
MIDI inviabile a qualsiasi DAW (Ableton, ecc.) tramite una porta MIDI virtuale
(loopMIDI su Windows).

---

## Novità della versione 0.6.0

* **MIDI affidabile su sessioni lunghe.** Dopo uno STOP/START o un cambio di
  setup le voci parlano subito (prima potevano restare mute per l'intera
  durata della sessione precedente). Gli errori di invio verso loopMIDI non
  sono più silenziosi: la riga di stato mostra `sent`/`err`, e in caso di
  stallo l'app si **riconnette da sola** alla porta e pulisce le note appese.
* **"max silence"** (schede NOTES / CH): failsafe di vivacità — se una voce
  tace troppo a lungo ritrigghera comunque, così nessun canale MIDI "muore"
  quando la banda è stabile. Default 10 s, 0 = OFF.
* **Il simulatore dorme davvero.** DOG: ciclo ~20 min con fase REM che
  accelera l'attività (theta/beta/gamma, twitch); HUMAN: ciclo ~90 min.
  Umano e cane ora evolvono e si distinguono a orecchio.

Le versioni precedenti sono documentate nel [CHANGELOG](CHANGELOG.md).

---

## Avvio rapido (build portabile Windows)

1. Estrai la cartella `SOGNO_CANE`.
2. Doppio click su **START.bat**.
3. Per inviare MIDI a una DAW installa
   [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html), crea una
   porta, poi premi **REFRESH** → **CONNECT** in SOGNO_CANE.

## Avvio da sorgente (sviluppo)

```bash
pip install -e .            # oppure: pip install -r requirements.txt
python -m sogno_cane        # GUI
python -m sogno_cane --selftest   # test headless della pipeline
pytest                      # suite di test
```

---

## Come funziona

```
 sorgente            analisi              strategie               MIDI
┌──────────┐   ┌──────────────────┐   ┌────────────────┐   ┌──────────────┐
│ simulatore│→ │ finestra mobile  │→ │ per-canale note │→ │ MidiOutput   │
│  oppure   │   │ + band power FFT │   │ CC, soglie,    │   │ (rtmidi)     │
│ file EEG  │   │ (Hann + Welch)   │   │ Markov, clip   │   │ → loopMIDI   │
└──────────┘   └──────────────────┘   └────────────────┘   └──────────────┘
```

* **`eeg/`** — simulatore (profili HUMAN/DOG distinti), pacchetti Unicorn a 17
  colonne, estrazione band power.
* **`midi/`** — motore di mappatura, scale musicali, *pacing* (`pacing.py`) e le
  strategie in `midi/strategies/`.
* **`core/`** — `engine.py` (thread realtime), `sources.py` (simulatore o
  riproduzione da array).
* **`io/`** — `edf.py` (lettore EDF/BDF), `loaders.py` (auto-detect formato),
  `recording.py` + `archive.py` (registrazione e archivio).
* **`ui/`** — interfaccia PySide6 (schede STUDIO e ARCHIVE).

## Mappatura: strategie

| Strategia            | Cosa produce                                   |
|----------------------|------------------------------------------------|
| Per-channel band     | una voce melodica per canale EEG               |
| Per-channel CC       | un controllo continuo (CC) per canale          |
| Threshold trigger    | eventi/accordi al superamento di una soglia    |
| Coherence CC         | CC dalla coerenza tra due gruppi di canali     |
| Markov generative    | melodia generativa pilotata dalle bande        |
| Clip launcher        | note brevi per lanciare clip in Ableton        |

Ogni strategia è regolabile dalle schede di mappatura (banda, scala, tonica,
canale MIDI, intervallo/durata/soglia per la sparsità musicale).

## Distribuzione e aggiornamento automatico

L'app può **aggiornarsi da sola** quando il PC è online: all'avvio controlla un
manifest JSON che pubblichi tu, scarica la nuova versione (con verifica
SHA-256) e la applica al riavvio tramite `update_apply.py` (in modo sicuro,
senza toccare file in uso). C'è anche un pulsante **UPDATES** per il controllo
manuale.

- **Far scaricare l'eseguibile da altri PC**: comprimi la cartella portabile e
  caricala su un host con link pubblico (consigliato **GitHub Releases**, gratis).
- **Pubblicare un aggiornamento**: alza la versione in
  [sogno_cane/__init__.py](sogno_cane/__init__.py), esegui
  `python tools/build_release.py --url <url> --out dist` (genera
  `sogno_cane_package.zip` + `version.json` con checksum) e caricali come asset
  di una Release.
- **Dove guarda l'app**: imposta `DEFAULT_UPDATE_URL` in
  [sogno_cane/update.py](sogno_cane/update.py) (o la variabile
  `SOGNO_CANE_UPDATE_URL`, o `update_url` in settings.json).

Istruzioni complete passo-passo: [portable/AGGIORNAMENTI.txt](portable/AGGIORNAMENTI.txt).

## Dove sono i dati

Registrazioni e impostazioni stanno in `~/.sogno_cane/` (oppure nella cartella
indicata dalla variabile d'ambiente `SOGNO_CANE_HOME`). Gli aggiornamenti
scaricati stanno in `~/.sogno_cane/updates/`.

## Licenza

MIT.
