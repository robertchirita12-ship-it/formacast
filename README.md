# FormaCast — server care se actualizează singur

Model Dixon-Coles antrenat pe **rezultate reale** (full-time + repriza întâi) și **cornere reale**,
de pe football-data.co.uk. Reîmprospătare automată la fiecare 6 ore. Un singur serviciu servește
și API-ul (JSON) și aplicația (o pagină).

## Ce face diferit față de versiunea din telefon (artifact)
- Rulează de pe un **server**, deci descarcă singur datele (fără blocajul 429/CORS din browser).
- Antrenează modelul pe **forma reală** a echipelor, nu pe inversarea cotelor → poți vedea **edge** real.
- Prima repriză din **golurile reale de la pauză**, cornerele din **istoricul real de cornere**.
- Se **actualizează singur** — mereu meciurile de azi înainte, fără să-mi ceri mie reîmprospătarea.

## Fișiere
- `main.py` — backend (FastAPI): fetch + model + API + servește pagina.
- `index.html` — aplicația (React din CDN), se conectează la API.
- `requirements.txt` — dependințe.

## Rulare locală (pe calculator)
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Deschizi `http://localhost:8000`. Prima încărcare a datelor durează 1–3 minute
(descarcă și antrenează în fundal). Vezi starea la `http://localhost:8000/api/health`.

## Lansare online gratuit (Render.com — merge și din browserul telefonului)
1. Fă-ți cont pe github.com și pune aceste 3 fișiere într-un repository nou
   (butonul „Add file → Upload files").
2. Cont pe render.com → **New → Web Service** → conectează repository-ul.
3. Setări:
   - Environment: **Python 3**
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Plan: **Free**
4. Deploy. În ~2 min ai un link public (ex: `https://formacast.onrender.com`).

> Pe planul gratuit Render adoarme serviciul după inactivitate; prima accesare după pauză
> durează ~30s până pornește. Pentru „mereu treaz" există planuri plătite ieftine.

## Reglaje (în `main.py`)
- `LEAGUES` — ce ligi și ce sezoane descarci (mai multe sezoane = model mai stabil).
- `HALF_LIFE_DAYS` — cât de mult contează forma recentă vs. cea veche (implicit 180).
- `REFRESH_HOURS` — cât de des se reîmprospătează (implicit 6).
- `MARGIN` — marja adăugată la cotele estimate (piețele fără cotă reală).

## API
- `GET /api/health` — stare + ultima actualizare.
- `GET /api/leagues` — ligile disponibile.
- `GET /api/fixtures?div=E0` — meciurile viitoare cu toate piețele, cote și edge.
- `GET /api/backtest?div=E0&edge=0.05` — validare walk-forward (acuratețe, ROI, curbă de profit).
  Rulează în fundal; reapelează până când `status` devine `done`.

## Funcții incluse
- **Tab Backtest** în aplicație: alegi liga + pragul de edge, vezi acuratețea reală și randamentul istoric.
- **Auto-ping**: serverul se auto-accesează la 10 min ca să nu adoarmă (folosește `RENDER_EXTERNAL_URL`,
  setat automat de Render). Îl poți opri cu variabila de mediu `KEEP_ALIVE=0`.

## Login
Pagina are o blocare simplă pe email+parolă (stocate ca hash SHA-256 în `index.html`).
E o barieră cosmetică, **nu securitate reală** — oricine are fișierele poate ocoli.
Nu folosi aici parola de la email; pune una unică.

## Notă
Instrument de analiză, nu sfat financiar. Cotele estimate sunt aproximative.
Selecțiile din același meci sunt corelate. Pariază responsabil.
