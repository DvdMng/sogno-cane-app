# SOGNO_CANE

**EEG → MIDI** per uomo e cane (g.tec Unicorn Hybrid Black), con simulatore
integrato, registrazione/archivio dei segnali e riproduzione di file EEG.

SOGNO_CANE genera (o legge) flussi EEG a 8 canali, ne estrae la potenza nelle
bande cerebrali (delta / theta / alpha / beta / gamma) e le traduce in musica
MIDI inviabile a qualsiasi DAW (Ableton, ecc.) tramite una porta MIDI virtuale
(loopMIDI su Windows).

---

## Collegare un EEG vero (Unicorn BCI Core)

L'app legge le cuffie **Unicorn BCI Core** in tempo reale attraverso lo
stream di rete del **Unicorn BCI Core Recorder** di g.tec.

0. **Una volta sola:** la *g.tec Suite 2024* installa solo un launcher con le
   licenze. Aprilo, inserisci License Key ed e-mail ricevute da g.tec con
   l'acquisto, premi *Activate* e scarica il **Unicorn BCI Core Recorder**.
1. Nel Recorder: *Connect* alla cuffia → `Settings` → `Network` →
   *Sending data via UDP* (IP `127.0.0.1`, porta **1001**) → *Play*.
   Se è attivo solo LSL, attiva anche UDP. Filtri: meglio spenti — l'app
   toglie da sola offset degli elettrodi e ronzio a 50 Hz.
2. In SOGNO_CANE premi **EEG DAL VIVO** sul pannello: porta e formato si
   riconoscono da soli.
3. Con **due** cuffie (umano + cane): due Recorder, anche sulla **stessa**
   porta 1001 — l'app li distingue dal mittente e ne dà uno per pannello.
   EEG DAL VIVO su entrambi i pannelli; se risultano invertiti, **SCAMBIA EEG**.

Verifica rapida senza aprire l'app:

```bash
python -m sogno_cane --live-test
```

Formati riconosciuti automaticamente: **testo** del BCI Core Recorder (righe
CSV `EXG 1..8, SATURATION, CNT, BAT, VALID, DT, TRIG`, anche con colonne
disattivate) e **binario** a 17 float dell'Unicorn UDP Interface (Hybrid
Black). Le registrazioni **CSV** e **BDF+** del Recorder si riproducono con
**FILE**. Guida completa: `portable/EEG_COLLEGAMENTO.txt`.

## Novità della versione 0.9.3

* **L'EEG dal vivo scorre fluido e senza ritardo che cresce.** La cuffia
  consegna i campioni via Bluetooth a pacchetti di 4 ogni ~16 ms; l'app li
  ritagliava in blocchi da 100 ms sul proprio orologio (tracciato a scatti,
  10 movimenti al secondo) e, poiché la cuffia va a ~250,65 Hz, accumulava
  ritardo minuto dopo minuto. Ora i campioni passano appena arrivano, il
  tracciato scorre a 60 fotogrammi al secondo e il ritardo resta fermo.
* **Frequenza misurata alla fonte** nella riga di stato del pannello
  ("EEG dal vivo 250 Hz", in rosa se il Recorder ne manda meno) e in
  `--live-test`.

## Novità della versione 0.9.1

* **Due cuffie, due segnali distinti.** Con due BCI Core Recorder attivi
  l'app leggeva un solo EEG e lo mostrava su entrambi i pannelli. Ora ogni
  pannello si prende il suo Recorder — anche se entrambi trasmettono sulla
  stessa porta predefinita — e non copia mai il segnale dell'altro.
* **SCAMBIA EEG**: se uomo e cane risultano invertiti, li scambia all'istante.

## Novità della versione 0.9.0

* **Pronto per le cuffie Unicorn BCI Core vere**, verificato sulle
  registrazioni ufficiali del dispositivo e sul codice dei ricevitori g.tec:
  legge il formato testo del BCI Core Recorder, toglie da solo l'offset
  degli elettrodi (il dispositivo è accoppiato in DC) e il ronzio di rete a
  50 Hz, ricuce i campioni persi dal Bluetooth.
* **EVENTI e CLIP adattivi**: scattano quando la banda sale molto sopra la
  sua norma recente, qualunque sia la cuffia — prima, con l'EEG vero,
  restavano mute.
* **Forma d'onda e file del Recorder** corretti con i dati reali;
  diagnostica `--live-test` che mostra offset, segnale e ronzio per
  elettrodo.

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
