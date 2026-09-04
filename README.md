# SOGNO_CANE

**EEG → MIDI** per uomo e cane (g.tec Unicorn Hybrid Black), con simulatore
integrato, registrazione/archivio dei segnali e riproduzione di file EEG.

SOGNO_CANE genera (o legge) flussi EEG a 8 canali, ne estrae la potenza nelle
bande cerebrali (delta / theta / alpha / beta / gamma) e le traduce in musica
MIDI inviabile a qualsiasi DAW (Ableton, ecc.) tramite una porta MIDI virtuale
(loopMIDI su Windows).

---

## Collegare un EEG vero (Unicorn)

L'app legge un **Unicorn** in tempo reale attraverso lo stream di rete che gli
strumenti g.tec già forniscono — non serve installare nulla in più.

1. Accendi il dispositivo e avvia **uno** di questi (entrambi inclusi con la
   g.tec Suite):
   * **Unicorn BCI Core Recorder** → `Settings` → `Network` →
     *Sending data via UDP* (porta predefinita **1001**), oppure
   * **Unicorn Suite** → `DevTools` → `UnicornUDP.exe` → imposta IP
     `127.0.0.1` e una porta, poi `Open` → `Start`.
2. In SOGNO_CANE premi **LIVE EEG** sul pannello del dispositivo: la porta e
   il formato dei canali vengono riconosciuti da soli.
3. Con **due** EEG (umano + cane) usa **una porta per ciascuno** — per esempio
   1001 per l'umano e 1002 per il cane — e premi LIVE EEG su entrambi i
   pannelli. Ogni dispositivo ha il suo motore e la sua porta MIDI.

Verifica rapida senza aprire l'app:

```bash
python -m sogno_cane --live-test
```

Formati riconosciuti automaticamente: 17 canali (Unicorn UDP Interface,
identico al formato interno dell'app), 14 canali (BCI Core Recorder con tutte
le colonne) e il caso di soli 8 canali EEG. Le registrazioni **CSV** e
**BDF+** del Recorder si riproducono con **LOAD FILE**.

## Novità della versione 0.8.0

* **Interfaccia in italiano** e riordinata secondo l'ordine reale delle
  operazioni: profilo → COLLEGA il MIDI → scegli la SORGENTE → AVVIA.
* **Versione sempre visibile** (titolo e barra in alto) e finestrella
  **"cosa è cambiato"** al primo avvio dopo un aggiornamento.
* **Riga di salute a colori**: dice sempre se il segnale entra e se il MIDI
  esce. Avviare senza porta MIDI non è più un silenzio misterioso.
* **Sorgente come controllo segmentato** (SIMULATORE / FILE / EEG DAL VIVO),
  **pannelli UMANO e CANE distinti** da banda e colore, **monitor MIDI**
  ridimensionabile che lascia spazio alla forma d'onda.

## Novità della versione 0.7.0

* **EEG reali in diretta**: nuovo pulsante **LIVE EEG** con rilevamento
  automatico di porta e formato; due dispositivi in parallelo su due porte.
  Un'interruzione del segnale non ferma il motore e il flusso riprende da
  solo. Nessuna dipendenza aggiuntiva (solo UDP di sistema).
* **`--live-test`**: diagnostica da riga di comando per capire in dieci
  secondi se i dati arrivano e su quale porta.

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
