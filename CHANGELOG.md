# Changelog

## 0.9.1 — due cuffie, due segnali distinti
Segnalazione dal campo: con due sessioni del BCI Core Recorder attive (uomo
e cane), SOGNO_CANE leggeva un solo EEG e lo mostrava uguale su entrambi i
pannelli.

**Diagnosi** (riprodotta con due Recorder simulati e i veri pannelli): l'app
identificava un flusso EEG **solo dal numero di porta UDP**. Così:
- due Recorder sulla stessa porta — ed è il caso normale, perché ogni
  Recorder nasce su 1001 — finivano **mescolati campione per campione** in un
  solo pannello;
- l'opzione `SO_REUSEADDR` permetteva a due pannelli di ascoltare la stessa
  porta: su Windows il secondo restava muto, oppure — con l'invio in
  broadcast — riceveva **la copia esatta del primo** (lo specchio);
- la ricerca automatica del secondo pannello riproponeva, e preselezionava, il
  segnale già usato dal primo.
Su 5 configurazioni realistiche la 0.9.0 ne sbagliava 4.

**Soluzione:**
- un solo ricevitore per porta (`UdpHub`), in **uso esclusivo**
  (`SO_EXCLUSIVEADDRUSE`): nessun altro socket può più condividerla di
  nascosto;
- i dati vengono smistati per **mittente**: ogni Recorder trasmette dal suo
  indirizzo, e **ogni pannello si prende un solo Recorder**. Funziona anche
  con entrambi i Recorder sulla porta predefinita, senza toccare le
  impostazioni;
- la ricerca elenca ogni segnale con chi lo sta già usando e **non offre mai
  al secondo pannello il segnale del primo**; se trasmette un solo Recorder,
  il secondo pannello lo dice chiaramente invece di copiarlo;
- se un Recorder riavvia la trasmissione (cambia indirizzo di invio), il suo
  pannello lo ritrova da solo, senza toccare l'altro;
- nuovo pulsante **SCAMBIA EEG**: se uomo e cane risultano invertiti si
  scambiano all'istante (i dati in rete non dicono quale cuffia sia del
  cane: lo sa solo chi le indossa). Il dialogo di scelta mostra la batteria
  di ciascun segnale, da confrontare con la finestra del Recorder;
- la porta 1000 non viene più sondata: è quella su cui il Recorder *ascolta*
  i trigger;
- `--live-test` misura ogni cuffia separatamente, fianco a fianco.

Verificato con due registrazioni reali del BCI Core inviate insieme sulla
stessa porta 1001: ciascun pannello riceve la sua, con i contatori dei
campioni tutti consecutivi (nessuna mescolanza), e il MIDI reale esce da
entrambi. Test: +13, **216 totali**.

## 0.9.0 — pronto per le cuffie vere: verificato sui dati reali del BCI Core
Studiando la g.tec Suite 2024 arrivata con i dispositivi, il codice sorgente
ufficiale dei ricevitori g.tec e le registrazioni pubbliche fatte col
Unicorn BCI Core, è emerso che la 0.8.0 **non era compatibile** con il
dispositivo vero. Qui la correzione di ogni punto, verificata facendo
scorrere quelle registrazioni reali nell'app.

- **Il Recorder trasmette testo, non numeri binari.** Il ricevitore UDP
  ufficiale di g.tec fa solo `Encoding.ASCII.GetString(...)`: ogni
  datagramma è una riga CSV (`EXG 1..8, SATURATION, CNT, BAT, VALID, DT,
  TRIG`, virgola come separatore, punto decimale). La 0.7.0 aveva esteso al
  Recorder il formato binario a 17 float dell'Unicorn UDP Interface
  (Hybrid Black): con il Recorder vero l'app avrebbe scartato quasi tutto.
  Ora testo e binario si riconoscono da soli, anche più righe per pacchetto,
  CRLF, padding, intestazioni, variante con punto e virgola.
- **Offset elettrico (DC).** Il BCI Core è accoppiato in DC: ogni elettrodo
  arriva con 20.000–77.000 µV di offset che deriva. Sul dato reale il
  vecchio calcolo dava **100% delta e 0% a tutto il resto**: la musica
  avrebbe seguito l'offset, non il cervello. Ora ogni finestra viene
  privata di offset e deriva lineare prima dello spettro.
- **Ronzio di rete a 50 Hz.** Con elettrodi a secco era il **100% della
  banda gamma**. Ora i 50 Hz e le armoniche sono esclusi da tutte le bande
  (configurabile: `MappingConfig.mains_hz`). Risultato sui dati reali:
  a occhi chiusi l'alfa è il 52% della potenza posteriore, a occhi aperti
  il 15% — l'effetto Berger, chiarissimo (rapporto 8:1).
- **Buchi del Bluetooth.** Il dispositivo marca i campioni persi con
  VALID = 0, e la 0.7.0 durante un'interruzione passava al motore degli
  zeri: accanto a 50.000 µV di offset è un gradino enorme che fa scattare
  tutte le voci insieme. Ora si tiene l'ultimo campione valido, sia dal
  vivo sia rileggendo i file.
- **EVENTI e CLIP tornano a suonare.** Le soglie fisse (80 e 60 µV²) erano
  tarate sul simulatore: sul dato reale EVENTI scattava **0 volte al
  minuto** (anzi, non era mai scattata nemmeno col simulatore) e CLIP 1,6.
  Ora sono **adattive**: un evento parte quando la banda sale di N
  "deviazioni tipiche" sopra la propria norma recente — lo stesso risultato
  su simulatore, cuffie a secco o con gel, umano o cane. Tarate sui dati
  reali: CLIP ~16/min (la densità che si sentiva già col simulatore),
  EVENTI ~12/min. La soglia fissa in µV² resta disponibile togliendo la
  spunta "adattiva". Le vecchie configurazioni passano da sole al modo
  adattivo.
- **Forma d'onda visibile con i dati reali.** Disegnava i valori grezzi:
  con l'offset le tracce finivano fuori schermo. Ora il display toglie
  offset e ronzio di rete (filtro causale, nessun salto all'avvio).
- **File del Recorder letti correttamente.** I CSV e BDF del Recorder
  venivano letti come 14 "canali" (anche contatore, batteria, trigger…) e
  funzionavano solo perché l'EEG viene per primo. Ora si tengono solo gli
  8 EXG e i campioni persi vengono ricuciti. Il lettore BDF è stato
  verificato su un file reale: coincide col CSV entro 0,02 µV.
- **Diagnostica `--live-test` più utile:** dice da quale programma g.tec
  arrivano i dati, l'offset di ogni elettrodo, l'ampiezza EEG senza
  ronzio (così si riconosce un elettrodo con cattivo contatto) e quanti
  campioni si perdono.
- **Guida `EEG_COLLEGAMENTO.txt` riscritta** sulla procedura reale:
  attivazione della licenza nel launcher g.tec, download del Recorder,
  impostazioni di rete, filtri consigliati, due cuffie.
- Test: +30 (formato reale del Recorder, offset e rete, campioni persi,
  soglie adattive e indipendenza dalla scala, file del Recorder, display)
  — **203 totali**.

## 0.8.0 — interfaccia leggibile, in italiano, che dice la verità
Nasce da un problema concreto: dopo aver installato la 0.7.0 non si capiva
se fosse cambiato qualcosa. Non era un difetto dell'aggiornamento — era
l'interfaccia che non lo diceva.

- **La versione è sempre visibile**: nel titolo della finestra e accanto al
  pulsante AGGIORNAMENTI. Prima si scopriva solo cliccando.
- **"Cosa è cambiato" dopo ogni aggiornamento**: all'avvio successivo appare
  una finestrella con versione e note di rilascio. Le note venivano già
  scaricate e poi buttate via; ora `update_apply.py` le lascia in
  `updates/last_applied.json` e l'app le mostra una volta sola.
- **Avviare senza MIDI non è più un silenzio inspiegabile**: lo stato dice
  "NESSUNA PORTA MIDI — non uscirà alcun suono" e la riga di salute lo
  ripete. Niente finestra modale: bloccare l'utente durante una
  performance sarebbe peggio del problema.
- **Riga di salute leggibile e a colori**: prima 8pt grigio spento, ora
  monospaziata e a colpo d'occhio — verde quando entra segnale ed esce
  MIDI, rosa quando qualcosa non va (nessun dato EEG, MIDI scollegato,
  invii falliti), grigia da fermo.
- **COLLEGA è un richiamo all'azione** (giallo) finché il MIDI non è
  collegato, poi torna un pulsante qualsiasi: la gerarchia rispecchia
  l'ordine reale delle operazioni.
- **Le tre sorgenti sono un controllo segmentato** (SIMULATORE / FILE /
  EEG DAL VIVO): quella attiva è illuminata, non più da indovinare leggendo
  un'etichetta. Se annulli la scelta del file o la connessione fallisce, il
  controllo torna a dire la verità invece di mentire.
- **UMANO e CANE si distinguono a colpo d'occhio**: banda colorata a
  sinistra e titolo ciano/arancio. Prima erano gemelli identici.
- **Il monitor ATTIVITÀ MIDI non ruba più spazio**: era fisso a 120px anche
  da vuoto; ora sta in un divisorio trascinabile e parte piccolo, lasciando
  la scena alla forma d'onda.
- **Tutta l'interfaccia è in italiano** — pulsanti, stati, messaggi, aiuto,
  schede di mappatura e archivio.
- Test: +10 sulle nuove garanzie UX (visibilità versione, avviso di
  aggiornamento a colpo singolo, coerenza del controllo segmentato, avviso
  MIDI non bloccante) — **173 totali**.

## 0.7.0 — EEG veri, in diretta
Fino alla 0.6.0 l'app sapeva solo **simulare** un EEG o **rileggere un file**:
non esisteva alcun percorso di acquisizione dal vivo (nessun socket, nessuna
seriale, nessun LSL nel codice). Con l'arrivo degli **Unicorn BCI Core** —
il modello che g.tec vende esplicitamente per registrazioni su umani, cani e
cavalli — questa versione aggiunge la lettura in tempo reale e la inietta nel
flusso di traduzione già collaudato, senza toccarlo.

- **Nuova sorgente LIVE (UDP)** — `sogno_cane/eeg/live.py`. L'app non parla
  col Bluetooth: consuma lo stream di rete che gli strumenti g.tec già
  emettono (Unicorn BCI Core Recorder → Network → *Sending data via UDP*,
  oppure Unicorn Suite → DevTools → `UnicornUDP.exe`). Zero dipendenze nuove:
  solo la libreria standard, quindi il bundle portabile resta invariato.
- **Formato riconosciuto da solo.** Il datagramma dell'Unicorn UDP Interface
  è di 68 byte = 17 float32 (`EEG1-8, ACC XYZ, GYR XYZ, BAT, CNT, VALID`):
  **coincide byte per byte** con il pacchetto che l'app usa dal primo giorno,
  quindi passa senza conversione. Il BCI Core Recorder invia invece le sue
  colonne (`EEG1-8, SAT, CNT, BAT, VALID, DT, STATUS`): l'app riconosce il
  layout dalla dimensione del datagramma e mappa batteria/contatore/validità
  al posto giusto, senza mai scambiare una flag di saturazione per un canale
  EEG.
- **Plug and play**: il pulsante **LIVE EEG** su ogni dispositivo ascolta le
  porte note, trova da solo quella attiva e si collega. Se ne trova più di
  una chiede quale; se non ne trova nessuna spiega cosa avviare e su quale
  porta. Due dispositivi = due porte (es. 1001 umano, 1002 cane), ognuno col
  proprio motore e la propria uscita MIDI.
- **Robustezza in scena** (nello spirito della 0.6.0): un buco sul filo non
  ferma più niente — il motore riceve silenzio e **riprende da solo** quando
  il segnale torna, mentre lo stato dice chiaramente *"LIVE EEG — silent for
  Ns (is the g.tec sender running?)"*. Il backlog è limitato: se il consumo
  rallenta si scartano i campioni vecchi invece di accumulare ritardo.
- **Diagnostica da riga di comando**: `python -m sogno_cane --live-test`
  cerca lo stream, dice porta, layout, campioni ricevuti e ampiezza di picco.
  Serve a rispondere in dieci secondi alla domanda "arriva qualcosa?".
- **Nota sui file**: i CSV registrati dal BCI Core Recorder erano già
  riproducibili con LOAD FILE (le 8 colonne EEG stanno in testa e finiscono
  correttamente nel motore); restano leggibili anche i BDF+.
- Test: +16 dedicati (layout dei tre mittenti, dropout, backlog limitato,
  datagrammi corrotti, rilevamento porta, pannello Qt, CLI) — **163 totali**.

## 0.6.0
Risponde ai feedback del 29/08: "quando cambio setup o fermo/riavvio
l'encefalogramma la comunicazione MIDI si interrompe, a volte un canale
smette di trasmettere, e l'app dice che manda dati mentre loopMIDI è
impallato"; "umano e cane sembrano identici, il ciclo del sonno del cane
(~20 min) non accelera mai in fase REM".

- **Fix: MIDI muto dopo STOP/START o cambio setup.** Le strategie di
  mappatura sopravvivono al motore (la UI riusa gli stessi oggetti), ma il
  nuovo motore riparte con l'orologio a zero: i timestamp di pacing della
  sessione precedente bloccavano le voci **per una durata pari all'intera
  sessione precedente** (10 minuti suonati = 10 minuti di silenzio dopo il
  riavvio, per voce). Ora l'avvio del motore azzera lo stato di runtime di
  ogni strategia (note ricordate, gate, throttle) e, per difesa in
  profondità, i gate si auto-riparano se rilevano l'orologio tornato
  indietro. Al riavvio le voci parlano subito e i CC ri-emettono subito il
  valore corrente (Ableton si riallinea senza aspettare una variazione).
- **Fix: "l'app manda ma loopMIDI è fermo".** Gli errori di invio rtmidi
  non vengono più inghiottiti: ora sono contati e visibili nella riga di
  stato di ogni dispositivo (`sent N` / `err N`). Dopo 8 invii falliti
  consecutivi l'app dichiara **MIDI STALLED** e **si riconnette da sola**
  alla stessa porta per nome (utile quando loopMIDI ricrea la porta o il
  driver si inceppa); alla riconnessione invia All-Notes-Off su tutti i 16
  canali per pulire eventuali note appese, e lo stato mostra
  "(reconnected)".
- **Fix: "uno dei nove canali smette di trasmettere".** La soglia di
  cambiamento (isteresi) poteva ammutolire per sempre una voce se la banda
  restava stabile. Nuovo controllo **"max silence"** (schede NOTES / CH):
  se una voce tace più di N secondi con segnale udibile, ritrigghera
  comunque (anche sulla stessa nota). Default **10 s**, 0 = OFF
  (comportamento precedente). Salvato nelle configurazioni, retrocompatibile.
- **Novità: architettura del sonno nel simulatore.** I profili HUMAN e DOG
  ora dormono davvero: il CANE cicla ogni **~20 minuti** con una coda REM
  (~5 min) in cui theta/beta/gamma salgono, le fluttuazioni accelerano
  (più note, più densità) e i "twitch" muscolari aumentano — il sogno si
  sente; l'UMANO segue il ciclo classico da **~90 minuti** con
  un'atonia REM più composta. I due stream divergono subito e non suonano
  più identici. I profili senza architettura del sonno restano stazionari
  come prima.

## 0.5.0
- **Durata delle note (gate) finalmente controllabile** — fix del feedback
  "la nota MIDI resta lunga ~2 secondi e blocca gli inviluppi". Prima il
  note_off partiva solo al *prossimo* evento (cambio di nota o fine frase):
  con gli intervalli di pacing di default una nota restava aperta 1.5–2.5 s,
  e con la soglia di cambiamento poteva restare aperta indefinitamente
  (suono continuo, inviluppi del synth bloccati).
- Nuovo controllo **"note length"** (in millisecondi) nelle schede
  NOTES / CH, MARKOV ed EVENTS: default **15 ms** — la nota diventa un
  trigger breve e gli inviluppi di Ableton attaccano/rilasciano liberamente.
  Impostando **0 = HOLD (legato)** si ritrova il comportamento precedente
  (nota tenuta fino al cambio di altezza).
- Il note_off temporizzato è inviato da uno **scheduler dedicato con
  precisione al millisecondo** (`NoteOffScheduler` in `core/engine.py`),
  perché il ciclo di analisi a 100 ms è troppo grossolano per note da 15 ms.
  Le note pendenti vengono rilasciate a STOP / fine file (niente note appese).
- Il parametro è salvato/caricato con le configurazioni (SAVE/LOAD CONFIG)
  ed è retrocompatibile: le config 0.3/0.4 caricano col default 15 ms.
- Robustezza (da review): un re-trigger della stessa nota con gate lungo non
  viene più troncato dal note_off precedente (supersessione nello scheduler);
  lo scheduler sopravvive a stop/start del motore; nessun thread/memoria
  trapelato a fine riproduzione file; contatore eventi thread-safe.

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
