# Data4Kids — MVP

Interaktywna aplikacja edukacyjna w Streamlit dla dzieci 7–14: misje, quizy, album naklejek, XP i poziomy.

## Szybki start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Struktura repo
```
data4kids/
├─ app.py
├─ requirements.txt
├─ README.md
└─ .streamlit/           # (opcjonalnie)
   └─ config.toml
```

## Konfiguracja (opcjonalnie)
Plik `.streamlit/config.toml`:
```toml
[server]
headless = true

[theme]
base = "light"
```

## Deploy na Streamlit Community Cloud
1. Wrzucić pliki (`app.py`, `requirements.txt`, `README.md`) do publicznego repo na GitHubie.
2. Wejść na https://share.streamlit.io → **New app**.
3. Wybrać repo/branch i wskazać `app.py` jako główny plik.
4. Po zbudowaniu dostaniesz adres w stylu `https://twoja-apka.streamlit.app`.

## Uwagi
- Pierwsze uruchomienie w chmurze trwa dłużej (pobieranie pakietów).
- Dla dużych CSV zalecane są małe próbki (setki KB–kilka MB).
- Wersje pakietów trzymamy w bezpiecznych widełkach; w razie potrzeby zablokujemy je „na sztywno”.
