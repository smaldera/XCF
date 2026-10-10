# Procedura: collegare il sensore Amptek SDD a WSL e usarlo con AmptekControl.py

Dispositivo di riferimento: **Amptek Digital Pulse Processor / Silicon Labs DP5**, VID:PID `10c4:842a`, busid tipico `1-5` (verificare con `usbipd list`, il busid può cambiare in base alla porta USB usata).

Distro WSL: **Ubuntu-20.04**
Cartella dei file su WSL: `/home/xcf/amptekhardwareinterface/` (contiene `AmptekControl.py`, `TestSDD.py`, ecc.)

---

## 1. [ADMIN — una tantum] Condividi (bind) il dispositivo USB

Il bind va fatto **una sola volta**: resta persistente tra riavvii (compare come `Persisted` in `usbipd list`).

Apri PowerShell **come amministratore** (tasto destro → Esegui come amministratore):

```powershell
usbipd list
```

Individua il busid del dispositivo Amptek (es. `1-5`), poi:

```powershell
usbipd bind --busid 1-5
```

> Se `usbipd list` restituisce già `Shared` per il dispositivo, il bind è già attivo e questo passaggio si può saltare.

> Per la regola di permessi USB dentro WSL (anche questa da fare una volta sola), vedi **Appendice A**.

---

## 2. [NON-ADMIN] Apri una sessione WSL e lasciala aperta

Apri una **normale** finestra PowerShell (utente standard, prompt tipo `C:\Users\XCF>`, **non** amministratore) e digita:

```powershell
wsl
```

**Non chiudere questa finestra**: deve restare aperta per tutta la sessione di lavoro. Se tutte le finestre WSL si chiudono, la VM WSL2 si spegne e va tutto rifatto da capo (modulo incluso).

Questo terminale verrà chiamato **terminale 1** nei passi seguenti.

---

## 3. [NON-ADMIN, dentro WSL — terminale 1] Carica il modulo vhci_hcd

```bash
sudo modprobe vhci_hcd
```

Nessun output = successo. (Serve rifarlo ogni volta che WSL viene riavviato da zero con `wsl --shutdown`, perché i moduli caricati non sopravvivono al riavvio della VM.)

Verifica che sia effettivamente caricato:

```bash
ls --directory /sys/bus/platform/drivers/vhci_hcd
```

Se il percorso viene stampato senza errori, il modulo è attivo.

---

## 4. [NON-ADMIN] Attacca il dispositivo a WSL

Apri una **seconda** finestra PowerShell **non amministratore** (**terminale 2**), lasciando il terminale 1 aperto in background:

```powershell
usbipd attach --wsl --busid 1-5
```

Dovrebbe completarsi senza errori.

---

## 5. [NON-ADMIN, dentro WSL — terminale 1] Verifica che il dispositivo sia visibile

```bash
lsusb
```

Deve comparire una riga con `ID 10c4:842a` (Silicon Labs DP5 Digital Pulse Processor).

---

## 6. [NON-ADMIN, dentro WSL — terminale 1] Vai nella cartella del codice e testa la connessione

```bash
cd /home/xcf/amptekhardwareinterface/
python3 -c "from AmptekControl import SDD; s = SDD(); print(s.PingPong())"
```

Output atteso: `Ping:  True` seguito da `True`.

Se invece compare `failed opening usb connection: Access denied` → i permessi USB non sono ancora impostati, vedi **Appendice A** (va fatta una volta sola, poi non serve più ripeterla).

---

## 7. [NON-ADMIN, dentro WSL — terminale 1] Crea la cartella di output (se non esiste)

`np.savez` non crea da sola le cartelle mancanti: se il path scelto per `-path` non esiste ancora sul lato Windows, l'acquisizione va a buon fine ma il **salvataggio fallisce**. Prima di ogni nuova sottocartella:

```bash
mkdir -p "/mnt/c/Users/XCF/<percorso scelto>/"
```

esempio:

```bash
mkdir -p "/mnt/c/Users/XCF/Desktop/SaveNpz/test/"
```

---

## 8. [NON-ADMIN, dentro WSL — terminale 1] Lancia l'acquisizione

```bash
python3 TestSDD.py -path Desktop/SaveNpz/test/ -time 10 -name TestConnessione
```

- `-path` è relativo a `/mnt/c/Users/XCF/` (viene concatenato automaticamente da `TestSDD.py`).
- `-time` è il livetime in secondi.
- `-name` è il prefisso del nome file.

Il file `.npz` risultante compare in `C:\Users\XCF\<path>\<name>_<anno>_<mese>_<giorno>_<ora>_<minuto>.npz` e si può rileggere con `ReadNpzTest.py`.

> Nota: senza sorgente/fascio davanti al sensore lo spettro sarà piatto/vuoto — è normale, non un errore. Serve solo a confermare che tutta la catena software↔hardware funziona.

---

## Appendice A — [NON-ADMIN, dentro WSL] Regola udev per i permessi USB (una tantum)

Da fare **una sola volta** (finché non cambi distro WSL o resetti le regole udev). Necessaria solo se il passo 6 dà `Access denied`.

1. Crea il file di regola:
   ```bash
   sudo nano /etc/udev/rules.d/99-amptek.rules
   ```
   contenuto:
   ```
   SUBSYSTEM=="usb", ATTR{idVendor}=="10c4", ATTR{idProduct}=="842a", MODE="0666"
   ```
   salva ed esci (Ctrl+O, invio, Ctrl+X in nano).

2. Ricarica le regole:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

3. Ri-attacca il dispositivo per far applicare la regola sul nuovo evento di "aggiunta" (da PowerShell non-admin su Windows):
   ```powershell
   usbipd detach --busid 1-5
   usbipd attach --wsl --busid 1-5
   ```

4. Riprova il test del passo 6 **senza sudo**. Se funziona, la regola è attiva in modo permanente e non va più rifatta.

---

## Note generali

- Il **bind** (passo 1) e la **regola udev** (Appendice A) sono operazioni **una tantum**.
- Il **modprobe** (passo 3) e l'**attach** (passo 4) vanno ripetuti ogni volta che:
  - riavvii il PC,
  - fai `wsl --shutdown`,
  - scolleghi/ricolleghi fisicamente il dispositivo USB.
- Se `usbipd attach` fallisce con *"There is no WSL 2 distribution running"* → il terminale 1 (passo 2) non è aperto o è stato chiuso: riapri una sessione WSL e lasciala attiva.
- Se `usbipd attach` fallisce con *"WSL kernel is not USBIP capable"* → assicurati di aver fatto il passo 3 (modprobe) **prima** dell'attach, nella stessa sessione WSL attiva.
- Se `TestSDD.py` fallisce al salvataggio con `FileNotFoundError` → manca la cartella di destinazione, vedi passo 7.
