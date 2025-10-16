# Data4Kids — uruchamianie lokalne (PowerShell)
# Uruchamiaj:  kliknij PPM → "Uruchom z PowerShell", albo:  .\run.ps1

# 1) Virtualenv
python -m venv .venv

# 2) Aktywacja środowiska
.\.venv\Scripts\Activate.ps1

# 3) Aktualizacja pip i instalacja zależności
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Start aplikacji
streamlit run app.py

# Jeśli system blokuje skrypty, uruchom jako Administrator:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
