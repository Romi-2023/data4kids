
import os
import json
import hashlib
import secrets
import random
import io
import re
from math import ceil
from datetime import datetime, date
from dateutil import tz
from typing import Optional, List, Dict
from fpdf import FPDF
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import psycopg2

APP_NAME = "Data4Kids"
VERSION = "0.9.0"

DONATE_BUYCOFFEE_URL = os.environ.get(
    "D4K_BUYCOFFEE_URL",
    "https://buycoffee.to/data4kids"  # TODO: podmień na swój prawdziwy link
)

DONATE_PAYPAL_URL = os.environ.get(
    "D4K_PAYPAL_URL",
    "paypal.me/RomanKnopp726"  # albo link z przycisku PayPal Donate
)

DONATE_BANK_INFO = os.environ.get(
    "D4K_BANK_INFO",
    (
        "Odbiorca: Roman Knopp / data4kids\n"
        "Nr konta: PL17 1140 2004 0000 3702 7712 0566\n"
        "Bank: mBank\n"
        "Tytuł: Darowizna na rozwój Data4Kids"
    )
)
# ---------------------------------
# Utilities & basic security (MVP)
# ---------------------------------
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Storage paths
# Statyczne dane (quizy, zadania, lektury itp.) trzymamy w katalogu "data" obok app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Te pliki służą tylko jako fallback lokalny (np. przy dev), na produkcji używamy bazy
USERS_FILE = os.path.join(DATA_DIR, "users.json")
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")
DONORS_FILE = os.path.join(DATA_DIR, "donors.json")   # zgłoszenia do konkursów (lokalnie)
DRAWS_FILE = os.path.join(DATA_DIR, "draws.json")     # historia losowań (lokalnie)

# --- BAZA DANYCH (PostgreSQL przez psycopg2) do trwałego przechowywania JSON-ów ---
DATABASE_URL = os.environ.get("DATABASE_URL")  # ustawiane automatycznie przez DigitalOcean


def get_db_connection():
    """Zwraca połączenie z bazą lub None jeśli brak DATABASE_URL."""
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)


def ensure_kv_table():
    """Tworzy tabelę kv_store, jeśli jeszcze nie istnieje."""
    if not DATABASE_URL:
        return
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )
    finally:
        conn.close()


def kv_get_json(key: str, default):
    """Odczyt JSON-a spod klucza z bazy; jeśli brak/blad – zwraca default."""
    if not DATABASE_URL:
        return default
    conn = get_db_connection()
    if conn is None:
        return default
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM kv_store WHERE key = %s", (key,))
                row = cur.fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return default
    except Exception:
        return default
    finally:
        conn.close()


def kv_set_json(key: str, value) -> None:
    """Zapis JSON-a pod kluczem w bazie (UPSERT)."""
    if not DATABASE_URL:
        return
    payload = json.dumps(value, ensure_ascii=False)
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO kv_store (key, value)
                    VALUES (%s, %s)
                    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
                    """,
                    (key, payload),
                )
    finally:
        conn.close()


# Upewnij się przy starcie, że tabela istnieje
ensure_kv_table()

def _load_donors():
    # 1. Próba odczytu z bazy
    records = kv_get_json("donors", None)
    if records is not None:
        return records

    # 2. Fallback lokalny (dev) z pliku JSON
    if not os.path.exists(DONORS_FILE):
        return []
    try:
        with open(DONORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_donors(records: list) -> None:
    # 1. Zapis do bazy (jeśli dostępna)
    kv_set_json("donors", records)

    # 2. Opcjonalny zapis lokalny (np. przy dev)
    try:
        with open(DONORS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        # Na produkcji zapis do pliku może się nie udać – ignorujemy
        pass

def _load_draws():
    records = kv_get_json("draws", None)
    if records is not None:
        return records

    if not os.path.exists(DRAWS_FILE):
        return []
    try:
        with open(DRAWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_draws(records: list) -> None:
    kv_set_json("draws", records)

    try:
        with open(DRAWS_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _load_users():
    # 1. Odczyt z bazy
    db = kv_get_json("users", None)
    if db is not None:
        return db

    # 2. Fallback z pliku (np. lokalnie)
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(db: dict) -> None:
    # 1. Zapis do bazy
    kv_set_json("users", db)

    # 2. Próba zapisu do pliku (może się nie udać na produkcji – pomijamy błąd)
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

from typing import Optional

def get_admin_totp_secret() -> Optional[str]:
    """Sekret TOTP admina z bazy (kv_store)."""
    data = kv_get_json("admin_totp_secret", None)
    if isinstance(data, dict):
        return data.get("secret")
    if isinstance(data, str):
        return data
    return None

def set_admin_totp_secret(secret: str) -> None:
    """Zapis sekretu TOTP admina do bazy."""
    kv_set_json("admin_totp_secret", {"secret": secret})


# === Parent PIN helpers (persistent in users.json) ===
def _ensure_parent_pin_record():
    db = _load_users()
    if "_parent_pin" not in db:
        salt = secrets.token_hex(16)
        db["_parent_pin"] = {"salt": salt, "hash": hash_text(salt + "1234")}
        _save_users(db)
    return _load_users()

def get_parent_pin_record():
    db = _ensure_parent_pin_record()
    rec = db.get("_parent_pin", {})
    return rec.get("salt", ""), rec.get("hash", "")

def verify_parent_pin(pin: str) -> bool:
    salt, h = get_parent_pin_record()
    return hash_text(salt + str(pin)) == h

def set_parent_pin(new_pin: str):
    if not new_pin.isdigit() or len(new_pin) < 4:
        raise ValueError("PIN musi mieć co najmniej 4 cyfry.")
    db = _ensure_parent_pin_record()
    salt = secrets.token_hex(16)
    db["_parent_pin"] = {"salt": salt, "hash": hash_text(salt + new_pin)}
    _save_users(db)

def hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()

def save_progress():
    if "user" in st.session_state and st.session_state.user:
        db = _load_users()
        u = st.session_state.user
        if u in db:
            db[u]["xp"] = st.session_state.xp
            db[u]["stickers"] = sorted(list(st.session_state.stickers))
            db[u]["badges"] = sorted(list(st.session_state.badges))
            _save_users(db)


# Age groups & levels
AGE_GROUPS = {"7-9": (7, 9), "10-12": (10, 12), "13-14": (13, 14)}
LEVEL_THRESHOLDS = [0, 30, 60, 100]  # L1:0+, L2:30+, L3:60+, L4:100+

def age_to_group(age: Optional[int]) -> str:
    if age is None:
        return "10-12"
    for label, (lo, hi) in AGE_GROUPS.items():
        if lo <= age <= hi:
            return label
    return "10-12"

def current_level(xp: int) -> int:
    if xp >= 100: return 4
    if xp >= 60: return 3
    if xp >= 30: return 2
    return 1

# -----------------------------
# Demo datasets (vary by age)
# -----------------------------
FAV_FRUITS = ["jabłko", "banan", "truskawka", "winogrono", "arbuz"]
FAV_ANIMALS = ["kot", "pies", "zebra", "słoń", "lama", "delfin"]
COLORS = ["czerwony", "zielony", "niebieski", "żółty", "fioletowy"]
CITIES = ["Warszawa", "Kraków", "Gdańsk", "Wrocław"]

@st.cache_data(show_spinner=False)
def make_dataset(n: int, cols: List[str], seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    data = {}
    if "wiek" in cols:
        data["wiek"] = [random.randint(7, 14) for _ in range(n)]
    if "wzrost_cm" in cols:
        data["wzrost_cm"] = [round(random.gauss(140, 12), 1) for _ in range(n)]
    if "ulubiony_owoc" in cols:
        data["ulubiony_owoc"] = [random.choice(FAV_FRUITS) for _ in range(n)]
    if "ulubione_zwierze" in cols:
        data["ulubione_zwierze"] = [random.choice(FAV_ANIMALS) for _ in range(n)]
    if "ulubiony_kolor" in cols:
        data["ulubiony_kolor"] = [random.choice(COLORS) for _ in range(n)]
    if "wynik_matematyka" in cols:
        data["wynik_matematyka"] = [max(0, min(100, int(random.gauss(70, 15)))) for _ in range(n)]
    if "wynik_plastyka" in cols:
        data["wynik_plastyka"] = [max(0, min(100, int(random.gauss(75, 12)))) for _ in range(n)]
    if "miasto" in cols:
        data["miasto"] = [random.choice(CITIES) for _ in range(n)]
    return pd.DataFrame(data)

DATASETS_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "7-9": {
        "Łatwy (mały)": ["wiek", "ulubiony_owoc", "miasto"],
        "Łatwy+ (z kolorem)": ["wiek", "ulubiony_owoc", "ulubiony_kolor", "miasto"],
    },
    "10-12": {
        "Średni": ["wiek", "wzrost_cm", "ulubiony_owoc", "miasto"],
        "Średni+": ["wiek", "wzrost_cm", "ulubiony_owoc", "ulubione_zwierze", "miasto"],
    },
    "13-14": {
        "Zaawansowany": ["wiek", "wzrost_cm", "wynik_matematyka", "wynik_plastyka", "miasto", "ulubiony_owoc"],
        "Zaawansowany+": ["wiek", "wzrost_cm", "wynik_matematyka", "wynik_plastyka", "miasto", "ulubiony_owoc", "ulubione_zwierze"],
    },
}

# -----------------------------
# UI style
# -----------------------------
KID_EMOJI = "🧒🎈📊"
PARENT_EMOJI = "🔒👨‍👩‍👧"

st.set_page_config(
    page_title=f"{APP_NAME} — MVP",
    page_icon="📚",
    layout="wide",
    menu_items={"About": f"{APP_NAME} v{VERSION} — MVP"},
)

st.markdown(
    """
    <style>
      .big-title {font-size: 2.2rem; font-weight: 800;}
      .muted {color: #6b7280;}
      .pill {display:inline-block;padding:.15rem .55rem;border-radius:9999px;background:#EEF2FF;font-size:.8rem;margin-left:.3rem}
      .kid {background:#DCFCE7}
      .parent {background:#FEF9C3}
      .badge {display:inline-block;margin:.25rem;padding:.25rem .5rem;border-radius:.8rem;background:#F0F9FF;border:1px solid #bae6fd}
      .sticker {display:flex;align-items:center;gap:.5rem;padding:.6rem;border-radius:.8rem;border:1px dashed #cbd5e1;margin:.3rem 0}
      .locked {opacity:.35;filter:grayscale(100%)}
    </style>
    """,
    unsafe_allow_html=True,
)

# Stickers (unchanged)
STICKERS: Dict[str, Dict[str, str]] = {
    "sticker_bars": {"emoji": "📊", "label": "Mistrz Słupków", "desc": "Poprawny wykres słupkowy."},
    "sticker_points": {"emoji": "🔵", "label": "Mistrz Punktów", "desc": "Poprawny wykres punktowy."},
    "sticker_detect": {"emoji": "🍉", "label": "Arbuzowy Tropiciel", "desc": "Zadanie detektywistyczne z arbuzem."},
    "sticker_sim": {"emoji": "🎲", "label": "Badacz Symulacji", "desc": "Symulacja rzutu monetą."},
    "sticker_clean": {"emoji": "🩺", "label": "Doktor Danych", "desc": "Naprawianie literówek."},
    "sticker_story": {"emoji": "📖", "label": "Opowieściopisarz", "desc": "Fabuła piknikowa."},
    "sticker_hawkeye": {"emoji": "👁️", "label": "Oko Sokoła", "desc": "Quiz obrazkowy — spostrzegawczość."},
    "sticker_math": {"emoji": "➗", "label": "Mat-fun", "desc": "Zadanie z matematyki wykonane!"},
    "sticker_polish": {"emoji": "📝", "label": "Językowa Iskra", "desc": "Polski — części mowy/ortografia."},
    "sticker_history": {"emoji": "🏺", "label": "Kronikarz", "desc": "Historia — oś czasu."},
    "sticker_geo": {"emoji": "🗺️", "label": "Mały Geograf", "desc": "Geografia — stolice i kontynenty."},
    "sticker_physics": {"emoji": "⚙️", "label": "Fiz-Mistrz", "desc": "Fizyka — prędkość = s/t."},
    "sticker_chem": {"emoji": "🧪", "label": "Chemik Amator", "desc": "Chemia — masa molowa."},
    "sticker_english": {"emoji": "🇬🇧", "label": "Word Wizard", "desc": "Angielski — słówka/irregulars."},
    "sticker_german": {"emoji": "🇩🇪", "label": "Deutsch-Star", "desc": "Niemiecki — pierwsze poprawne zadanie."},
    "sticker_bio": {"emoji": "🧬", "label": "Mały Biolog", "desc": "Biologia — podstawy komórki i łańcucha pokarmowego."},
}

# Session state
defaults = {
    "parent_unlocked": False,
    "kid_name": "",
    "age": None,
    "age_group": "10-12",
    "dataset_name": None,
    "data": make_dataset(140, DATASETS_PRESETS["10-12"]["Średni"], seed=42),
    "activity_log": [],
    "xp": 0,
    "badges": set(),
    "stickers": set(),
    "missions_state": {},
    "hall_of_fame": [],
    "last_quest": None,
    "todays": None,
    "kids_mode": True,
    "user": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def log_event(event: str):
    stamp = datetime.now(tz=tz.gettz("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.append({"time": stamp, "event": event})

def flatten_glossary(categories: dict) -> dict:
    flat = {}
    for cat, entries in categories.items():
        flat.update(entries)
    return flat

# --- English glossary TTS (browser SpeechSynthesis) ---
def tts_button_en(text: str, key: str):
    # Renders a small speaker button that uses browser SpeechSynthesis (no external API)
    import json as _json
    safe_text = _json.dumps(str(text))
    btn_id = f"tts_{key}"

    # Używamy tokenów, żeby nie walczyć z klamrami w f-stringach/format
    html = """
<button id="__BTN__" style="padding:4px 8px;border-radius:8px;border:1px solid #ddd;background:#F0F9FF;cursor:pointer">
  🔊 Wymów
</button>
<script>
const b = document.getElementById("__BTN__");
if (b) {
  b.onclick = () => {
    try {
      const u = new SpeechSynthesisUtterance(__TEXT__);
      u.lang = 'en-US';
      u.rate = 0.95;
      u.pitch = 1.0;
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    } catch (e) {}
  };
}
</script>
"""
    html = html.replace("__BTN__", btn_id).replace("__TEXT__", safe_text)
    components.html(html, height=40)



# === Daily fantasy data helpers ===
from datetime import date
import hashlib

def _day_seed(salt="data4kids"):
    txt = f"{date.today().isoformat()}::{salt}"
    return int(hashlib.sha256(txt.encode("utf-8")).hexdigest(), 16) % (2**32)

def pick_daily_sample(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    rs = np.random.RandomState(_day_seed("daily_sample"))
    idx = rs.choice(len(df), size=n, replace=False)
    return df.iloc[idx].copy()

FANTASY_CITIES = ["Krainogród", "Miodolin", "Zefiriada", "Księżycolas", "Wróżkowo", "Słonecznikowo", "Tęczomir", "Gwizdacz"]
FANTASY_FRUITS = ["smocze jabłuszko", "tęczowa truskawka", "kosmiczny banan", "fioletowa gruszka", "złoty ananas", "śnieżna jagoda"]
FANTASY_NAMES = ["Aurelka", "Kosmo", "Iskierka", "Nimbus", "Gaja", "Tygrys", "Mira", "Leo", "Fruzia", "Błysk", "Luna", "Kornik"]

def _map_choice(value: str, pool: list, salt: str) -> str:
    key = f"{value}|{date.today().isoformat()}|{salt}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return pool[h[0] % len(pool)]

def jitter_numeric_col(s: pd.Series, pct: float = 0.03, salt: str = "jitter") -> pd.Series:
    rs = np.random.RandomState(_day_seed(salt))
    noise = rs.uniform(low=1 - pct, high=1 + pct, size=len(s))
    out = s.astype(float).values * noise
    if "wiek" in s.name.lower():
        out = np.round(out).astype(int)
    return pd.Series(out, index=s.index)

def apply_fantasy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c: c.lower() for c in df.columns}
    for c in df.columns:
        name = cols_lower[c]
        if "miasto" in name or "city" in name:
            df[c] = df[c].astype(str).apply(lambda v: _map_choice(v, FANTASY_CITIES, "city"))
        if "owoc" in name or "fruit" in name:
            df[c] = df[c].astype(str).apply(lambda v: _map_choice(v, FANTASY_FRUITS, "fruit"))
        if "imię" in name or "imie" in name or "name" in name:
            df[c] = df[c].astype(str).apply(lambda v: _map_choice(v, FANTASY_NAMES, "name"))
        if pd.api.types.is_numeric_dtype(df[c]):
            if any(k in name for k in ["wzrost", "cm", "waga", "kg", "height", "mass"]):
                df[c] = jitter_numeric_col(df[c], pct=0.03, salt=f"jitter:{c}")
            elif "wiek" in name or "age" in name:
                pass
    return df

def _is_count_choice(val: str) -> bool:
    return val == "count()"

# Global helpers for missions
def award(ok: bool, xp_gain: int, badge: Optional[str] = None, mid: str = ""):
    if ok:
        prev_done = st.session_state.missions_state.get(mid, {}).get("done", False)
        if not prev_done:
            st.session_state.xp += xp_gain
            if badge:
                st.session_state.badges.add(badge)
        st.session_state.missions_state[mid] = {"done": True}
    else:
        st.session_state.missions_state[mid] = {"done": False}
    save_progress()

def get_leaderboard(limit: int = 10, age_group: Optional[str] = None) -> List[Dict]:
    """Prosty ranking po XP – z opcjonalnym filtrem po grupie wiekowej."""
    db = _load_users()
    rows = []
    for name, profile in db.items():
        if name.startswith("_"):
            continue  # pomijamy rekordy techniczne

        group = profile.get("age_group")  # może być None dla starych profili
        if age_group and group != age_group:
            continue

        rows.append({
            "user": name,
            "xp": int(profile.get("xp", 0)),
            "badges": len(profile.get("badges", [])),
            "stickers": len(profile.get("stickers", [])),
            "age_group": group or "?"
        })
    rows.sort(key=lambda r: r["xp"], reverse=True)
    return rows[:limit]



def grant_sticker(code: str):
    if code in STICKERS: st.session_state.stickers.add(code)

def show_hint(mid: str, hint: str):
    key = f"hint_used_{mid}"
    if st.button("Podpowiedź 🪄 (-1 XP)", key=f"hintbtn_{mid}"):
        if not st.session_state.get(key, False):
            st.session_state.xp = max(0, st.session_state.xp - 1)
            st.session_state[key] = True
        st.caption(hint)

# Chemistry utilities
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007, "Na": 22.990, "Cl": 35.45}
def _molar_mass(formula: str) -> Optional[float]:
    import re
    tokens = re.findall(r"[A-Z][a-z]?\d*", formula)
    if not tokens: return None
    total = 0.0
    for tok in tokens:
        m = re.match(r"([A-Z][a-z]?)(\d*)", tok)
        if not m: return None
        el, num = m.group(1), m.group(2)
        if el not in ATOMIC_MASS: return None
        n = int(num) if num else 1
        total += ATOMIC_MASS[el] * n
    return total

# === MISSIONS (subset shown to keep file reasonable) ===
def mission_math_arith(mid: str):
    st.subheader("Matematyka ➗: szybkie działania")
    a, b = random.randint(2, 12), random.randint(2, 12)
    op = random.choice(["+", "-", "*"])
    true = a + b if op == "+" else (a - b if op == "-" else a * b)
    guess = st.number_input(f"Policz: {a} {op} {b} = ?", step=1, key=f"{mid}_g")
    if st.button(f"Sprawdź {mid}"):
        ok = (guess == true)
        award(ok, 6, badge="Szybkie liczby", mid=mid)
        if ok:
            grant_sticker("sticker_math")
            st.success("✅ Tak!")
        else:
            st.warning(f"Prawidłowo: {true}")
    show_hint(mid, "Pamiętaj: najpierw mnożenie, potem dodawanie/odejmowanie.")

def mission_math_line(mid: str):
    st.subheader("Matematyka 📈: prosta y = a·x + b")
    a = random.choice([-2, -1, 1, 2])
    b = random.randint(-3, 3)
    xs = list(range(-5, 6))
    df_line = pd.DataFrame({"x": xs, "y": [a*x + b for x in xs]})
    chart = alt.Chart(df_line).mark_line(point=True).encode(x="x:Q", y="y:Q")
    st.altair_chart(chart, use_container_width=True)
    q = st.radio("Jaki jest znak nachylenia a?", ["dodatni", "zerowy", "ujemny"], index=None, key=f"{mid}_slope")
    if st.button(f"Sprawdź {mid}"):
        sign = "zerowy" if a == 0 else ("dodatni" if a > 0 else "ujemny")
        ok = (q == sign)
        award(ok, 8, badge="Linia prosta", mid=mid)
        if ok:
            grant_sticker("sticker_math")
            st.success("✅ Dobrze!")
        else:
            st.warning("Podpowiedź: linia rośnie → dodatni; maleje → ujemny.")

def mission_polish_pos(mid: str):
    st.subheader("Język polski 📝: część mowy")
    sentence = "Ala ma kota i czerwony balon."
    st.write(f"Zdanie: _{sentence}_")
    pick = st.selectbox("Które słowo to rzeczownik?", ["Ala", "ma", "kota", "czerwony", "balon"], key=f"{mid}_pick")
    if st.button(f"Sprawdź {mid}"):
        ok = pick in {"Ala", "kota", "balon"}
        award(ok, 7, badge="Językowa Iskra", mid=mid)
        if ok:
            grant_sticker("sticker_polish")
            st.success("✅ Świetnie!")
        else:
            st.warning("Rzeczowniki to nazwy osób, rzeczy, zwierząt…")

# Map for mission IDs (if needed later)
def run_mission_by_id(mid: str):
    mapping = {
        "MAT-1": lambda: mission_math_arith("MAT-1"),
        "MAT-2": lambda: mission_math_line("MAT-2"),
        "POL-1": lambda: mission_polish_pos("POL-1"),
    }
    fn = mapping.get(mid)
    if fn: fn()
    else: st.info(f"(W przygotowaniu) {mid}")

# Helpers for tasks.json rotation
def get_today_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def days_since_epoch() -> int:
    return (date.today() - date(2025, 1, 1)).days

def generate_rower_certificate_pdf(username: str, date_str: str, correct: int, total: int, percent: int) -> bytes:
    """
    Tworzy certyfikat treningu karty rowerowej jako PDF i zwraca go jako bytes.
    Tytuł: DancingScript (jeśli dostępny), reszta: Arial (jeśli dostępny).
    """
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False, margin=0)
    pdf.add_page()

    # Ścieżki do czcionek
    script_path = os.path.join(BASE_DIR, "fonts", "DancingScript-VariableFont_wght.ttf")
    sans_path = os.path.join(BASE_DIR, "fonts", "arial.ttf")

    script_ok = False
    sans_ok = False

    # Najpierw próbujemy załadować Arial (Unicode)
    try:
        if os.path.exists(sans_path):
            pdf.add_font("Sans", "", sans_path, uni=True)
            sans_ok = True
    except Exception:
        sans_ok = False

    # Potem DancingScript – tylko do tytułu, też Unicode
    try:
        if os.path.exists(script_path):
            pdf.add_font("Script", "", script_path, uni=True)
            script_ok = True
    except Exception:
        script_ok = False

    # --- Tytuł ---
    title_text = "Certyfikat treningu – karta rowerowa"

    # Wybór fontu do tytułu
    if script_ok:
        pdf.set_font("Script", "", 34)
    elif sans_ok:
        pdf.set_font("Sans", "", 28)
    else:
        # Ostateczny fallback – Helvetica BEZ „–”, żeby nie wybuchło
        pdf.set_font("Helvetica", "B", 26)
        title_text = "Certyfikat treningu - karta rowerowa"

    # Ramka
    pdf.set_draw_color(200, 0, 80)
    pdf.set_line_width(1.5)
    pdf.rect(10, 10, 277, 190)

    # Obrazek odznaki (to co masz jako cert_bike.png)
    img_path = os.path.join(BASE_DIR, "assets", "cert_bike.png")
    if os.path.exists(img_path):
        # x, y, szerokość (dostosuj jeśli chcesz większy/mniejszy)
        pdf.image(img_path, x=20, y=22, w=40)

    # Tytuł
    pdf.set_xy(10, 20)
    pdf.cell(277, 15, title_text, align="C", ln=1)

    # --- Podtytuł i treść – używamy Arial jeśli się wczytał ---
    if sans_ok:
        pdf.set_font("Sans", "", 15)
    else:
        pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.ln(3)
    pdf.cell(277, 10, "Data4Kids – moduł przygotowania do karty rowerowej", align="C", ln=1)

    if sans_ok:
        pdf.set_font("Sans", "", 17)
    else:
        pdf.set_font("Helvetica", "", 15)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)
    pdf.multi_cell(
        0,
        10,
        txt=(
            f"Potwierdzamy, że {username} w dniu {date_str}\n"
            f"ukończył(a) egzamin próbny na kartę rowerową\n"
            f"z wynikiem {correct} / {total} ({percent}%)."
        ),
        align="C",
    )

    # Krótkie wyjaśnienie
    if sans_ok:
        pdf.set_font("Sans", "", 13)
    else:
        pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(90, 90, 90)
    pdf.ln(5)
    pdf.multi_cell(
        0,
        7,
        txt=(
            "Certyfikat dotyczy treningu w aplikacji Data4Kids i może być użyty jako "
            "potwierdzenie przygotowań dziecka do właściwego egzaminu na kartę rowerową."
        ),
        align="C",
    )

    # Miejsce na podpis rodzica + „pieczątka” systemu
    pdf.ln(18)
    if sans_ok:
        pdf.set_font("Sans", "", 12)
    else:
        pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)

    # lewa strona – podpis rodzica
    pdf.cell(138, 10, "......................................", align="C")
    pdf.cell(1)
    # prawa strona – tekst zamiast pustej linii
    pdf.cell(138, 10, "Potwierdzono w systemie Data4Kids", align="C", ln=1)

    pdf.cell(138, 6, "Opiekun / rodzic", align="C")
    pdf.cell(1)
    pdf.cell(138, 6, "(podpis elektroniczny systemu)", align="C", ln=1)


    # Stopka
    pdf.set_y(190)
    if sans_ok:
        pdf.set_font("Sans", "", 8)
    else:
        pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, "Wygenerowano automatycznie w Data4Kids – moduł 'Moja karta rowerowa'.", align="C")

    result = pdf.output(dest="S")  # w fpdf2 to jest bytes albo bytearray
    if isinstance(result, bytearray):
        pdf_bytes = bytes(result)
    else:
        pdf_bytes = result
    return pdf_bytes



def safe_load_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default
    
def load_glossary_all():
    """
    Wczytuje słowniczki z osobnych plików w data/glossary/*.json.

    Każdy plik ma postać:
    {
      "hasło": "definicja",
      "inne hasło": "inna definicja"
    }

    Nazwa pliku (bez .json), zamieniona na wielkie litery i spacje, jest nazwą zakładki,
    np. dane_i_statystyka.json -> "DANE I STATYSTYKA".
    """
    folder = os.path.join(DATA_DIR, "glossary")
    glossary = {}

    if not os.path.isdir(folder):
        return glossary

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(folder, fname)
        data = safe_load_json(path, default={})

        # matematyka.json -> "MATEMATYKA"
        # dane_i_statystyka.json -> "DANE I STATYSTYKA"
        base = os.path.splitext(fname)[0]
        subject_key = base.replace("_", " ").upper()

        if isinstance(data, dict):
            glossary[subject_key] = data

    return glossary

CATEGORIZED_GLOSSARY = load_glossary_all()


def load_tasks() -> Dict[str, list]:
    d = safe_load_json(TASKS_FILE, default={})
    if d:
        return d
    # fallback to top-level
    return safe_load_json('tasks.json', default={})

def pick_daily_chunk(task_list: list, k: int, day_index: int, subject: str) -> list:
    if not task_list:
        return []
    # Deterministyczny shuffle: zależny od przedmiotu, grupy i daty
    import hashlib, random
    seed_text = f"{subject}:{get_today_key()}"
    seed_int = int(hashlib.sha256(seed_text.encode('utf-8')).hexdigest(), 16) % (10**12)
    rng = random.Random(seed_int)
    shuffled = task_list[:]
    rng.shuffle(shuffled)
    if k <= 0:
        return []
    groups = ceil(len(shuffled) / k)
    idx = day_index % max(groups, 1)
    start = idx * k
    stop = start + k
    return shuffled[start:stop]

# ----- School tasks completion & XP helpers -----
def _task_id_from_text(text: str) -> str:
    return hashlib.sha256(("task::" + text).encode("utf-8")).hexdigest()[:12]

def _user_db_get(u: str):
    db = _load_users()
    return db.get(u)

def _user_db_set(u: str, profile: dict):
    db = _load_users()
    db[u] = profile
    _save_users(db)

def _get_today_completion_key() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def mark_task_done(user: str, subject: str, task_text: str, xp_gain: int = 5):
    profile = _user_db_get(user) or {}
    # ensure containers
    profile.setdefault("school_tasks", {})
    today = _get_today_completion_key()
    day_map = profile["school_tasks"].setdefault(today, {})
    subj_list = day_map.setdefault(subject, [])
    tid = _task_id_from_text(task_text)
    if tid not in subj_list:
        subj_list.append(tid)
        # award XP once
        st.session_state.xp += xp_gain
        save_progress()
    _user_db_set(user, profile)

def is_task_done(user: str, subject: str, task_text: str) -> bool:
    profile = _user_db_get(user)
    if not profile: return False
    today = _get_today_completion_key()
    tid = _task_id_from_text(task_text)
    try:
        return tid in profile.get("school_tasks", {}).get(today, {}).get(subject, [])
    except Exception:
        return False


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(f"<div class='big-title'>📚 {APP_NAME}</div>", unsafe_allow_html=True)
    st.caption("Misje, daily quest, symulacje, czyszczenie, fabuła, album, quizy, PRZEDMIOTY ✨")

    page = st.radio(
        "Przejdź do:",
        options=[
            "Start",
            "Poznaj dane",
            "Plac zabaw",
            "Misje",
            "Przedmioty szkolne",
            "Pomoce szkolne",
            "Quiz danych",
            "Quiz obrazkowy",
            "Album naklejek",
            "Słowniczek",
            "Moje osiągnięcia",
            "Hall of Fame",
            "Wsparcie & konkursy",
            "Regulamin",
            "Kontakt",
            "Administrator",
            "Panel rodzica",
        ],
    )
    st.checkbox("Tryb dziecięcy (prostszy widok)", key="kids_mode")

    with st.expander("Słowniczek (skrót)"):
        st.caption("Pełną listę pojęć znajdziesz w zakładce »Słowniczek«. 🔎")


    # --- Global fantasy mode toggle (sidebar) ---
    st.session_state.setdefault("fantasy_mode", True)
st.markdown("### 🌈 Tryb danych")
st.toggle("Fantastyczne nazwy + delikatny jitter", key="fantasy_mode")
def _try_unlock_parent():
    pin = st.session_state.get("parent_pin_input", "")
    if verify_parent_pin(pin):
        st.session_state["parent_unlocked"] = True
        st.session_state["parent_pin_input"] = ""
        st.success("Panel rodzica odblokowany.")
    else:
        st.session_state["parent_unlocked"] = False
        if pin:
            st.warning("Zły PIN. Spróbuj ponownie.")

# --- Globalny wymóg logowania dla stron dziecięcych ---
PUBLIC_PAGES = {"Start", "Regulamin", "Kontakt", "Administrator", "Panel rodzica", "Wsparcie & konkursy"}

if page not in PUBLIC_PAGES and not st.session_state.get("user"):
    st.info("Najpierw zaloguj się na stronie **Start**. Potem możesz korzystać z całej aplikacji. 🚀")
    st.stop()

# -----------------------------
# START (with auth gate)
# -----------------------------
if page == "Start":
    st.markdown("### 🔐 Logowanie")

    # Dwie kolumny: lewa = formularz, prawa = opis aplikacji
    col_form, col_info = st.columns([2, 1])

    with col_form:
        db = _load_users()

        # Jeśli użytkownik właśnie się zarejestrował, przełącz widok na "Zaloguj"
        # (robimy to ZANIM narysujemy st.radio)
        if st.session_state.get("just_registered"):
            st.session_state.auth_mode = "Zaloguj"
            st.session_state.reg_step = 1
            st.session_state.just_registered = False
            st.session_state.show_reg_success = True

        # --- jeśli NIKT nie jest zalogowany -> pokazujemy logowanie/rejestrację ---
        if not st.session_state.get("user"):
            # sterownik widoku: radio zamiast tabs
            if "auth_mode" not in st.session_state:
                st.session_state.auth_mode = "Zaloguj"

            auth_mode = st.radio(
                " ",
                ["Zaloguj", "Zarejestruj"],
                horizontal=True,
                key="auth_mode",
                label_visibility="collapsed",
            )

            # ---------- LOGOWANIE ----------
            if auth_mode == "Zaloguj":
                # jeżeli właśnie wróciliśmy po rejestracji – pokaż jednorazowy komunikat
                if st.session_state.get("show_reg_success"):
                    st.success("Utworzono konto! Teraz zaloguj się na swój login i hasło. 🎉")
                    st.session_state.show_reg_success = False

                li_user = st.text_input("Login", key="li_user")
                li_pass = st.text_input("Hasło", type="password", key="li_pass")
                if st.button("Zaloguj", key="login_btn"):
                    if li_user in db:
                        salt = db[li_user]["salt"]
                        if hash_pw(li_pass, salt) == db[li_user]["password_hash"]:
                            st.session_state.user = li_user
                            st.session_state.xp = int(db[li_user].get("xp", 0))
                            st.session_state.stickers = set(db[li_user].get("stickers", []))
                            st.session_state.badges = set(db[li_user].get("badges", []))
                            st.success(f"Zalogowano jako **{li_user}** 🎉")
                            st.rerun()  # po zalogowaniu chowamy panel logowania
                        else:
                            st.error("Błędne hasło.")
                    else:
                        st.error("Taki login nie istnieje.")

            # ---------- REJESTRACJA: 2 kroki ----------
            else:  # auth_mode == "Zarejestruj"
                # krok rejestracji: 1 = formularz, 2 = regulamin + potwierdzenie
                if "reg_step" not in st.session_state:
                    st.session_state.reg_step = 1

                re_user = st.text_input("Nowy login", key="reg_user")
                re_pass = st.text_input("Hasło", type="password", key="reg_pass")
                re_pass2 = st.text_input("Powtórz hasło", type="password", key="reg_pass2")

                # --- KROK 1: dane logowania ---
                if st.session_state.reg_step == 1:
                    st.caption("Krok 1/2: wpisz login i hasło, potem kliknij **Zarejestruj**.")

                    if st.button("Zarejestruj", key="reg_step1"):
                        # weryfikujemy podstawowe dane, ale JESZCZE nie tworzymy konta
                        login_pattern = r"^[A-Za-z0-9_-]{3,20}$"

                        if not re_user or not re_pass:
                            st.error("Podaj login i hasło.")
                        elif not re.match(login_pattern, re_user):
                            st.error(
                                "Login może zawierać tylko litery, cyfry, '-', '_' "
                                "i musi mieć od 3 do 20 znaków (bez spacji)."
                            )
                        elif len(re_pass) < 6:
                            st.error("Hasło musi mieć co najmniej 6 znaków.")
                        elif re_user in db:
                            st.error("Taki login już istnieje.")
                        elif re_pass != re_pass2:
                            st.error("Hasła się różnią.")
                        else:
                            st.session_state.reg_step = 2
                            st.success(
                                "Świetnie! Teraz przeczytaj Regulamin poniżej i potwierdź, "
                                "że się z nim zgadzasz (krok 2/2)."
                            )
                            st.rerun()

                # --- KROK 2: regulamin + zgoda ---
                elif st.session_state.reg_step == 2:
                    st.info(
                        "Krok 2/2: Regulamin Data4Kids – przeczytaj i zaznacz zgodę, aby założyć konto."
                    )

                    st.markdown(
                        """
                        #### 📜 Regulamin (skrót)

                        1. Dane służą tylko do działania aplikacji (logowanie, XP, misje), nie sprzedajemy ich i nie wysyłamy dalej.  
                        2. Nie wymagamy imienia i nazwiska ani maila – możesz używać pseudonimu.  
                        3. Hasła są haszowane, ale nadal dbaj o ich bezpieczeństwo i nie udostępniaj ich innym.  
                        4. Aplikacja ma charakter edukacyjny i może zawierać drobne błędy.  
                        5. Profil można w każdej chwili usunąć w Panelu rodzica.
                        """
                    )

                    accept = st.checkbox(
                        "Przeczytałem/przeczytałam i akceptuję regulamin Data4Kids.",
                        key="reg_accept_terms",
                    )

                    parent_ok = st.checkbox(
                        "Jestem w wieku 7 - 14 lat LUB rodzic/opiekun pomaga mi założyć konto.",
                        key="reg_parent_ok",
                    )

                    col_reg1, col_reg2 = st.columns([1, 1])
                    with col_reg1:
                        if st.button("⬅️ Wróć do edycji danych", key="reg_back"):
                            st.session_state.reg_step = 1
                            st.rerun()

                    with col_reg2:
                        if st.button("Akceptuję regulamin i zakładam konto ✅", key="reg_submit"):
                            if not accept:
                                st.error("Aby założyć konto, musisz zaakceptować regulamin.")
                            elif not parent_ok:
                                st.error(
                                    "Aby założyć konto, potrzebna jest zgoda rodzica/opiekuna "
                                    "lub potwierdzenie, że masz co najmniej 13 lat."
                                )
                            elif not re_user or not re_pass:
                                # na wszelki wypadek, gdyby ktoś odświeżył
                                st.error("Brakuje loginu lub hasła. Wróć do kroku 1.")
                            elif re_user in db:
                                st.error("Taki login już istnieje.")
                            elif re_pass != re_pass2:
                                st.error("Hasła się różnią. Wróć do kroku 1.")
                            else:
                                salt = secrets.token_hex(8)
                                db[re_user] = {
                                    "salt": salt,
                                    "password_hash": hash_pw(re_pass, salt),
                                    "xp": 0,
                                    "stickers": [],
                                    "badges": [],
                                    "accepted_terms_version": VERSION,
                                }
                                _save_users(db)
                                st.session_state.reg_step = 1
                                st.session_state.just_registered = True
                                st.rerun()

        # --- jeśli KTOŚ jest zalogowany -> mały status zamiast formularza ---
        else:
            st.success(f"Zalogowano jako **{st.session_state.user}** ✅")
            if st.button("Wyloguj", key="logout_btn"):
                st.session_state.user = None
                st.session_state.xp = 0
                st.session_state.badges = set()
                st.session_state.stickers = set()
                st.session_state.auth_mode = "Zaloguj"
                st.rerun()

    # Prawa kolumna: krótka informacja o aplikacji
    with col_info:
        st.markdown("### ℹ️ O aplikacji")
        st.write(
            """
            Aplikacja:

            - uczy konsekwentnego myślenia,
            - wzmacnia analizę informacji,
            - wspiera trening emocji,
            - tworzy automatyczne raporty dla rodzica,
            - dopasowuje misje do dziecka.

            Idealne dla edukacji domowej, szkół i zajęć indywidualnych.
            """
        )

    # --- dalej: tylko dla zalogowanego dziecka ---
    if not st.session_state.get("user"):
        st.info("Zaloguj się, aby kontynuować.")
        st.stop()

    # -------- Reszta ekranu Start --------
    st.markdown(
        f"<div class='big-title'>🧒 {KID_EMOJI} Witaj w {APP_NAME}!</div>",
        unsafe_allow_html=True,
    )
    colA, colB = st.columns([1, 1])
    with colA:
        # 1) Widget ma INNY klucz niż session_state.kid_name
        name_input = st.text_input("Twoje imię (opcjonalnie)", key="kid_name_input")
        raw_name = name_input.strip()

        # 2) Walidacja imienia: tylko litery, max 12 znaków
        name_pattern = r"^[A-Za-zÀ-ÖØ-öø-ÿ]{2,12}$"
        if raw_name:
            if re.match(name_pattern, raw_name):
                st.session_state.kid_name = raw_name
            else:
                st.warning(
                    "Imię może mieć tylko litery (bez spacji) i maks. 12 znaków. "
                    "Nie zapisuję tego imienia."
                )
                # jeżeli wcześniej nie było sensownego imienia, wyzeruj
                if not st.session_state.get("kid_name"):
                    st.session_state.kid_name = ""

        # 3) Jeśli dalej brak poprawnego imienia → generujemy pseudonim
        if not st.session_state.get("kid_name"):
            if "kid_nick" not in st.session_state:
                nick_roots = ["Lama", "Kometa", "Zorza", "Atlas", "Pixel", "Foka", "Błysk"]
                st.session_state.kid_nick = random.choice(nick_roots) + "-" + str(
                    random.randint(10, 99)
                )
            st.session_state.kid_name = st.session_state.kid_nick

        st.caption(f"Twój nick w aplikacji: **{st.session_state.kid_name}**")

        # 4) Reszta jak było
        age_in = st.number_input(
            "Ile masz lat?", min_value=7, max_value=14, step=1, value=10
        )
        st.session_state.age = int(age_in)
        st.session_state.age_group = age_to_group(int(age_in))
        group = st.session_state.age_group
        st.info(f"Twoja grupa wiekowa: **{group}**")

        presets = DATASETS_PRESETS[group]
        preset_name = st.selectbox("Wybierz zestaw danych", list(presets.keys()))
        st.session_state.dataset_name = preset_name
        if st.button("Załaduj zestaw danych"):
            cols = presets[preset_name]
            n = 100 if group == "7-9" else (140 if group == "10-12" else 180)
            st.session_state.data = make_dataset(
                n, cols, seed=random.randint(1, 999999)
            )
            st.success(f"Załadowano: {preset_name}")
            log_event(f"dataset_loaded_{group}_{preset_name}")

        if st.button("Start misji 🚀"):
            log_event(f"kid_started_{group}")
            st.success("Super! Wejdź do »Misje« i działamy.")

    with colB:
        st.write(
            """
            **Co zrobimy?**
            - Daily Quest ✅
            - Rysowanie, detektyw 🕵️
            - Symulacje 🎲, Czyszczenie ✍️, Fabuła 📖
            - Przedmioty szkolne 📚 (mat, pol, hist, geo, fiz, chem, ang)
            - Album naklejek 🗂️ i Quizy 🖼️🧠
            - XP, odznaki i poziomy 🔓, Hall of Fame 🏆
            """
        )
        st.markdown(
            f"XP: **{st.session_state.xp}** | Poziom: **L{current_level(st.session_state.xp)}** "
            + "".join(
                [f"<span class='badge'>🏅 {b}</span>" for b in st.session_state.badges]
            ),
            unsafe_allow_html=True,
        )


# -----------------------------
# Pozostałe podstrony (skrócone do kluczowych)
# -----------------------------
elif page == "Poznaj dane":
    st.markdown(
        f"<div class='big-title'>📊 {KID_EMOJI} Poznaj dane</div>",
        unsafe_allow_html=True,
    )

    df_base = st.session_state.data.copy()

    if df_base is None or len(df_base) == 0:
        st.info("Brak danych do eksploracji. Najpierw załaduj zestaw w zakładce Start.")
    else:
        fantasy_mode = st.session_state.get("fantasy_mode", True)

        # === 🎲 Eksperyment losowania próbek ===
        st.subheader("🎲 Eksperyment losowania próbek")

        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            sample_size = st.radio(
                "Wielkość próby (liczba osób):",
                [10, 50, 100],
                index=0,
                horizontal=True,
                help="Spróbuj różnych wielkości próby i zobacz, jak zachowuje się średnia."
            )
        with col_s2:
            st.caption(
                "Klikaj przycisk, aby wylosować nową próbkę tej samej wielkości. "
                "Dane pochodzą z tego samego zestawu."
            )

        max_n = len(df_base)
        if sample_size > max_n:
            sample_size = max_n

        if "sample_df" not in st.session_state:
            st.session_state["sample_df"] = None
            st.session_state["sample_size"] = sample_size

        if st.button(f"Wylosuj próbkę ({sample_size} osób)"):
            st.session_state["sample_size"] = sample_size
            st.session_state["sample_df"] = df_base.sample(
                n=sample_size,
                replace=False,
                random_state=random.randint(0, 10**9),
            )

        sample_df = st.session_state.get("sample_df")

        if isinstance(sample_df, pd.DataFrame) and not sample_df.empty:
            st.markdown("#### Podsumowanie wylosowanej próby")

            csa1, csa2, csa3 = st.columns(3)

            if "wiek" in sample_df.columns:
                mean_age = round(pd.to_numeric(sample_df["wiek"], errors="coerce").mean(), 2)
                csa1.metric("Średni wiek w próbie", mean_age)

            if "wzrost_cm" in sample_df.columns:
                mean_h = round(pd.to_numeric(sample_df["wzrost_cm"], errors="coerce").mean(), 1)
                csa2.metric("Śr. wzrost (cm) w próbie", mean_h)

            csa3.metric("Liczba osób w próbie", len(sample_df))

            if "wiek" in sample_df.columns:
                st.markdown("**Histogram wieku w próbie**")
                age_df = pd.DataFrame({"wiek": pd.to_numeric(sample_df["wiek"], errors="coerce")}).dropna()
                if not age_df.empty:
                    chart_age = (
                        alt.Chart(age_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("wiek:Q", bin=alt.Bin(maxbins=10), title="Wiek"),
                            y=alt.Y("count():Q", title="Liczba osób"),
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart_age, use_container_width=True)

            if "miasto" in sample_df.columns:
                st.markdown("**Ile osób z danego miasta?**")
                city_counts = (
                    sample_df["miasto"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "miasto", "miasto": "liczba"})
                )
                st.dataframe(city_counts, use_container_width=True)

            st.info(
                "Im większa próba, tym **stabilniejsza średnia** i rozkład – "
                "to właśnie proste Prawo wielkich liczb w praktyce. ✨"
            )
        else:
            st.caption(
                "Kliknij przycisk powyżej, aby wylosować pierwszą próbkę z danych."
            )

        st.divider()

        # === Zestaw dnia (mały wycinek danych do spokojnego oglądania) ===
        N = min(15, len(df_base))
        df_daily = pick_daily_sample(df_base, n=max(1, N)) if N else df_base

        df_view = apply_fantasy(df_daily) if fantasy_mode else df_daily

        st.subheader("📅 Zestaw dnia")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Liczba wierszy (zestaw dnia)", len(df_view))

        if "wiek" in df_view.columns:
            c2.metric(
                "Śr. wiek",
                round(pd.to_numeric(df_view["wiek"], errors="coerce").mean(), 1),
            )
        if "wzrost_cm" in df_view.columns:
            c3.metric(
                "Śr. wzrost (cm)",
                round(pd.to_numeric(df_view["wzrost_cm"], errors="coerce").mean(), 1),
            )
        if "miasto" in df_view.columns:
            c4.metric("Liczba miast", df_view["miasto"].nunique())

        with st.expander("Zobacz tabelę (zestaw dnia)"):
            st.caption(f"Zestaw dzienny: {date.today().isoformat()}")
            st.dataframe(df_view.head(50), use_container_width=True)

        st.divider()

        # === Analiza kolumn liczbowych ===
        st.subheader("📈 Kolumny liczbowe")

        num_cols = [c for c in df_view.columns if pd.api.types.is_numeric_dtype(df_view[c])]

        if num_cols:
            num_col = st.selectbox("Wybierz kolumnę do analizy:", num_cols)

            col_data = pd.to_numeric(df_view[num_col], errors="coerce").dropna()
            if not col_data.empty:
                desc = col_data.describe().to_frame().T
                st.markdown("**Statystyki opisowe:**")
                st.dataframe(desc, use_container_width=True)

                hist_df = pd.DataFrame({num_col: col_data})
                chart_hist = (
                    alt.Chart(hist_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=20), title=num_col),
                        y=alt.Y("count():Q", title="Liczba rekordów"),
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart_hist, use_container_width=True)
            else:
                st.caption("Brak danych w wybranej kolumnie.")
        else:
            st.caption("Brak kolumn liczbowych w tym zestawie.")

        st.divider()

        # === Analiza kolumn kategorycznych ===
        st.subheader("📊 Kolumny kategoryczne")

        cat_cols = [
            c
            for c in df_view.columns
            if df_view[c].dtype == "object" and df_view[c].nunique() > 1
        ]

        if cat_cols:
            cat_col = st.selectbox("Wybierz kolumnę kategoryczną:", cat_cols)

            vc = (
                df_view[cat_col]
                .value_counts()
                .reset_index()
                .rename(columns={"index": cat_col, cat_col: "liczba"})
            )

            st.markdown("**Najczęstsze wartości:**")
            st.dataframe(vc.head(10), use_container_width=True)

            chart_cat = (
                alt.Chart(vc.head(10))
                .mark_bar()
                .encode(
                    x=alt.X("liczba:Q", title="Liczba rekordów"),
                    y=alt.Y(f"{cat_col}:N", sort="-x", title=cat_col),
                )
                .properties(height=300)
            )
            st.altair_chart(chart_cat, use_container_width=True)
        else:
            st.caption(
                "Brak typowych kolumn kategorycznych (tekstowych) do analizy w tym zestawie."
            )

        st.divider()

        # === Korelacje między kolumnami liczbowymi ===
        st.subheader("🔗 Powiązania między kolumnami liczbowymi")

        if len(num_cols) >= 2:
            corr = df_view[num_cols].corr()
            corr_df = (
                corr.reset_index()
                .melt("index", var_name="kolumna2", value_name="korelacja")
                .rename(columns={"index": "kolumna1"})
            )

            chart_corr = (
                alt.Chart(corr_df)
                .mark_rect()
                .encode(
                    x=alt.X("kolumna2:N", title="Kolumna 2"),
                    y=alt.Y("kolumna1:N", title="Kolumna 1"),
                    color=alt.Color(
                        "korelacja:Q",
                        scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
                        title="korelacja",
                    ),
                    tooltip=["kolumna1", "kolumna2", "korelacja"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_corr, use_container_width=True)
            st.caption(
                "Korelacja bliska **1** oznacza silną dodatnią zależność, "
                "bliska **-1** – silną ujemną, a okolice **0** – brak wyraźnej zależności."
            )
        else:
            st.caption(
                "Do policzenia korelacji potrzeba co najmniej dwóch kolumn liczbowych."
            )

        st.divider()

        # === Prosty model liniowy: wiek vs wzrost ===
        st.subheader("📐 Prosty model liniowy (wiek → wzrost)")

        if "wiek" in df_view.columns and "wzrost_cm" in df_view.columns:
            reg_df = df_view[["wiek", "wzrost_cm"]].copy()
            reg_df["wiek"] = pd.to_numeric(reg_df["wiek"], errors="coerce")
            reg_df["wzrost_cm"] = pd.to_numeric(reg_df["wzrost_cm"], errors="coerce")
            reg_df = reg_df.dropna()

            if len(reg_df) >= 2:
                x = reg_df["wiek"].values
                y = reg_df["wzrost_cm"].values

                a, b = np.polyfit(x, y, 1)  # y ≈ a * wiek + b

                line_x = np.linspace(x.min(), x.max(), 50)
                line_y = a * line_x + b
                df_line = pd.DataFrame({"wiek": line_x, "wzrost_model": line_y})

                scatter = (
                    alt.Chart(reg_df)
                    .mark_circle(size=60, opacity=0.7)
                    .encode(
                        x=alt.X("wiek:Q", title="Wiek"),
                        y=alt.Y("wzrost_cm:Q", title="Wzrost (cm)"),
                        tooltip=["wiek", "wzrost_cm"],
                    )
                )

                line = (
                    alt.Chart(df_line)
                    .mark_line()
                    .encode(
                        x=alt.X("wiek:Q"),
                        y=alt.Y("wzrost_model:Q", title="Modelowany wzrost (cm)"),
                    )
                )

                st.altair_chart(scatter + line, use_container_width=True)
                st.caption(
                    "Kropki to dzieci z zestawu dnia, a linia to prosty model: "
                    "jak **średnio** rośnie wzrost wraz z wiekiem."
                )
            else:
                st.caption("Za mało danych, by narysować prostą regresji.")
        else:
            st.caption("Ten model wymaga kolumn „wiek” i „wzrost_cm” w danych.")

elif page == "Plac zabaw":
    st.markdown(f"<div class='big-title'>🧪 {KID_EMOJI} Plac zabaw z danymi</div>", unsafe_allow_html=True)
    df = st.session_state.data
    st.write("Wgraj swój plik CSV **albo** baw się gotowymi danymi.")
    uploaded = st.file_uploader("Wgraj CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.session_state.data = df_up
            st.success("Plik wgrany! Używamy Twoich danych.")
            log_event("csv_uploaded")
        except Exception as e:
            st.error(f"Błąd wczytywania CSV: {e}")
    base = st.session_state.data.copy()
    N = min(20, len(base)) if len(base) else 0
    df_daily = pick_daily_sample(base, n=max(1, N)) if N else base
    fantasy_mode = st.session_state.get("fantasy_mode", True)
    df_view = apply_fantasy(df_daily) if fantasy_mode else df_daily
    cols = st.multiselect("Kolumny do podglądu", df_view.columns.tolist(), default=df_view.columns[:4].tolist())
    st.caption(f"Zestaw dzienny: {date.today().isoformat()} • rekordów: {len(df_view)}")
    st.dataframe(df_view[cols].head(30), width='stretch')

elif page == "Misje":
    st.markdown(
        f"<div class='big-title'>🗺️ {KID_EMOJI} Misje</div>",
        unsafe_allow_html=True,
    )

    missions_path = os.path.join(DATA_DIR, "missions.json")
    missions = safe_load_json(missions_path, default=[])

    # ================================
    #  🔍 Analiza quizów -> profil trudności
    # ================================
    events = st.session_state.get("activity_log", [])
    quiz_profile = None
    hardest_areas = []
    area_labels = {}

    if events:
        dfq = pd.DataFrame(events)
        if "event" in dfq.columns:
            mask_quiz = dfq["event"].str.contains("quiz_ok") | dfq["event"].str.contains("quiz_fail")
            dfq = dfq[mask_quiz].copy()

            if not dfq.empty:
                def _parse_quiz_event(ev: str):
                    parts = str(ev).split("::")
                    if not parts or parts[0] not in ("quiz_ok", "quiz_fail"):
                        return None
                    status = "ok" if parts[0] == "quiz_ok" else "fail"
                    source = parts[1] if len(parts) > 1 else "inne"

                    # Domyślne wartości
                    category = source
                    area = source  # bardziej „czytelna” etykieta do misji

                    # Quiz danych:  quiz_ok::data::<qid>::<short_q>
                    if source == "data":
                        category = "data_quiz"
                        area = "Dane i liczby"

                    # Quiz obrazkowy:  quiz_ok::image::<cat>::<qid>::<short_q>
                    elif source == "image":
                        img_cat = parts[2] if len(parts) > 2 else "inne"
                        mapping = {
                            "emotions": "Emocje",
                            "shapes": "Kształty",
                            "plots": "Wykresy",
                            "objects": "Przedmioty",
                        }
                        area = mapping.get(img_cat, img_cat)
                        category = f"image_{img_cat}"

                    return {
                        "status": status,
                        "source": source,
                        "category": category,
                        "area": area,
                    }

                parsed = []
                for ev in dfq["event"]:
                    p = _parse_quiz_event(ev)
                    if p:
                        parsed.append(p)

                if parsed:
                    qdf = pd.DataFrame(parsed)
                    stats = (
                        qdf.groupby("area")
                        .agg(
                            total=("status", "size"),
                            wrong=("status", lambda s: (s == "fail").sum()),
                            ok=("status", lambda s: (s == "ok").sum()),
                        )
                        .reset_index()
                    )
                    stats["fail_pct"] = (
                        stats["wrong"] / stats["total"] * 100
                    ).round(1)

                    # Bierzemy tylko obszary, w których było przynajmniej kilka odpowiedzi
                    hardest_areas = (
                        stats[stats["total"] >= 3]
                        .sort_values(["fail_pct", "total"], ascending=[False, False])
                        .to_dict("records")
                    )

                    quiz_profile = stats

                    # mapka area -> (fail_pct, total)
                    for row in hardest_areas:
                        area_labels[row["area"]] = {
                            "fail_pct": row["fail_pct"],
                            "total": int(row["total"]),
                        }

    # ================================
    #  ⭐ Ranking misji wg profilu trudności
    # ================================
    recommended_ids = set()
    reasons_by_id = {}

    if missions and hardest_areas:
        # słowa kluczowe do dopasowania treści misji do obszaru trudności
        AREA_KEYWORDS = {
            "Emocje": ["emocje", "uczuc", "nastr", "smut", "strach", "rado", "złość"],
            "Kształty": ["kształt", "figura", "geometr", "trójkąt", "kwadrat", "koło"],
            "Wykresy": ["wykres", "diagram", "słupk", "liniow", "statystyk", "dane"],
            "Przedmioty": ["przedmiot", "rzecz", "otoczen", "obiekt"],
            "Dane i liczby": ["dane", "liczb", "procent", "średn", "tabela"],
        }

        # weźmy maksymalnie 3 „najtrudniejsze” obszary
        top_hard = hardest_areas[:3]

        for m in missions:
            mid = m.get("id") or m.get("title") or str(id(m))
            text = (m.get("title", "") + " " + m.get("desc", "")).lower()

            best_score = 0
            best_area = None

            for area_info in top_hard:
                area_name = area_info["area"]
                keys = AREA_KEYWORDS.get(area_name, [])
                # policz, ile słów kluczowych pasuje do tekstu misji
                score = sum(1 for kw in keys if kw and kw.lower() in text)
                if score > best_score:
                    best_score = score
                    best_area = area_name

            if best_score > 0 and best_area:
                recommended_ids.add(mid)
                meta = area_labels.get(best_area, {})
                fail_pct = meta.get("fail_pct")
                total = meta.get("total")
                if fail_pct is not None and total is not None:
                    reasons_by_id[mid] = (
                        f"Ta misja pasuje do obszaru, który jest teraz trudniejszy "
                        f"dla dziecka: **{best_area}** "
                        f"(błędnych odpowiedzi: {fail_pct}% z {total})."
                    )
                else:
                    reasons_by_id[mid] = (
                        f"Ta misja pasuje do obszaru, który wymaga teraz więcej ćwiczeń: "
                        f"**{best_area}**."
                    )

    # ================================
    #  UI: Misje rekomendowane + reszta
    # ================================
    if not missions:
        st.info("Brak misji. Dodaj je do data/missions.json")
    else:
        # --- Sekcja misji rekomendowanych ---
        st.markdown("### ⭐ Misje rekomendowane przez Data4Kids")

        if recommended_ids:
            for m in missions:
                mid = m.get("id") or m.get("title") or str(id(m))
                if mid not in recommended_ids:
                    continue

                with st.expander(f"🎯 {m.get('title','Misja')}"):
                    st.write(m.get("desc", ""))
                    st.caption("Kroki: " + ", ".join(m.get("steps", [])))

                    # wyjaśnienie, dlaczego misja jest polecana
                    reason = reasons_by_id.get(mid)
                    if reason:
                        st.markdown(
                            f"🧠 **Rekomendowane na podstawie quizów.**  \n{reason}"
                        )
                    else:
                        st.markdown(
                            "🧠 **Rekomendowane na podstawie quizów** – "
                            "pasuje do obszarów, w których dziecko popełnia więcej błędów."
                        )

                    if st.button(
                        "Zaznacz jako zrobioną 📝",
                        key=f"mis_rec_{mid}",
                    ):
                        st.success(
                            "Super! Zaznaczone jako zrobione (na razie bez przyznawania XP)."
                        )
                        try:
                            log_event(f"mission_done::{mid}::{m.get('title','')}")
                        except Exception:
                            pass
        else:
            # Brak danych lub brak dopasowanych misji
            st.caption(
                "Na razie brak specjalnych rekomendacji – "
                "potrzebujemy kilku odpowiedzi w quizach, żeby wiedzieć, "
                "co jest dla dziecka najtrudniejsze. 😊"
            )

        st.divider()

        # --- Wszystkie misje (w tym ewentualnie niepolecane) ---
        st.markdown("### 🌍 Wszystkie misje")

        for m in missions:
            mid = m.get("id") or m.get("title") or str(id(m))
            is_rec = mid in recommended_ids

            title_prefix = "🎯"
            if is_rec:
                title_prefix = "💡🎯"  # mały highlight również na liście ogólnej

            with st.expander(f"{title_prefix} {m.get('title','Misja')}"):
                st.write(m.get("desc", ""))
                st.caption("Kroki: " + ", ".join(m.get("steps", [])))

                if is_rec:
                    st.markdown(
                        "_Ta misja jest też na liście **rekomendowanych** na górze._"
                    )

                if st.button(
                    "Zaznacz jako zrobioną 📝",
                    key=f"mis_{mid}",
                ):
                    st.success(
                        "Super! Zaznaczone jako zrobione (na razie bez przyznawania XP)."
                    )
                    try:
                        log_event(f"mission_done::{mid}::{m.get('title','')}")
                    except Exception:
                        pass


elif page == "Quiz danych":
    st.markdown(f"<div class='big-title'>📊 {KID_EMOJI} Quiz danych</div>", unsafe_allow_html=True)
    dq_path = os.path.join(DATA_DIR, "quizzes", "data_quiz.json")
    dq = safe_load_json(dq_path, default={"items": []})
    items = dq.get("items", [])

    # --- dzienna rotacja pytań w Quizie danych ---
    all_items = items  # pełna baza pytań
    day_idx = days_since_epoch()
    k_daily = min(10, len(items))  # ile pytań dziennie – możesz zmienić np. na 15

    if items:
        # pick_daily_chunk losuje stałą (dla danego dnia) porcję pytań
        # i rotuje „kawałki” między kolejnymi dniami bez powtórek
        items = pick_daily_chunk(items, k_daily, day_idx, "data_quiz")

    st.caption(
        f"Dzisiejszy zestaw: {len(items)} pytań "
        f"(z {len(all_items)} w całej bazie)."
    )

    for i, t in enumerate(items, start=1):
        q = t["q"]
        opts = t["options"]
        corr = int(t["correct"])

        st.markdown(f"**{i}. {q}**")
        choice = st.radio(
            "Wybierz:",
            opts,
            key=f"dq_{i}",
            label_visibility="collapsed",
            index=None,
        )

        if st.button("Sprawdź ✅", key=f"dq_check_{i}"):
            if choice is None:
                st.warning("Wybierz odpowiedź.")
            else:
                # Przygotowanie skróconego opisu pytania + stabilnego ID
                try:
                    short_q = q if len(q) <= 60 else q[:57] + "..."
                    base = f"dq::{q}"
                    qid = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
                except Exception:
                    short_q = q[:60]
                    qid = None

                if opts.index(choice) == corr:
                    st.success("✅ Dobrze!")
                    # logujemy poprawną odpowiedź
                    try:
                        log_event(
                            f"quiz_ok::data::{qid or ''}::{short_q}"
                        )
                    except Exception:
                        pass
                else:
                    st.error(f"❌ Nie. Poprawna: **{opts[corr]}**.")
                    # logujemy błędną odpowiedź
                    try:
                        chosen = choice
                        correct_label = opts[corr]
                        log_event(
                            f"quiz_fail::data::{qid or ''}::{short_q}::{chosen}::{correct_label}"
                        )
                    except Exception:
                        pass


elif page == "Quiz obrazkowy":
    st.markdown(
        f"<div class='big-title'>🖼️ {KID_EMOJI} Quiz obrazkowy</div>",
        unsafe_allow_html=True,
    )

    iq_path = os.path.join(DATA_DIR, "quiz_images", "image_quiz.json")
    iq = safe_load_json(iq_path, default={"items": []})
    raw_items = iq.get("items", [])

    # --- Obsługa starego i nowego formatu --------------------------
    flat_items = []
    for item in raw_items:
        img = item.get("image")
        if not img:
            continue

        # nowe pole "category" (np. emotions, shapes, plots, objects)
        cat = item.get("category") or "inne"

        # stare pole "age_group" / "group" → filtrujemy wyżej w kodzie
        age = item.get("age_group") or item.get("group") or "10-12"

        flat_items.append(
            {
                "image": img,
                "q": item.get("q", ""),
                "options": item.get("options", []),
                "correct": item.get("correct", 0),
                "category": cat,
                "age_group": age,
            }
        )

    # --- Filtr po grupie wiekowej dziecka --------------------------
    age_label = st.session_state.age_group  # np. "8-10", "10-12"
    allowed_cats = iq.get("allowed_categories", ["emotions", "plots", "shapes", "objects"])

    age_items = [
        it for it in flat_items
        if (it.get("age_group") or "10-12") == age_label
        and (it.get("category") or "inne") in allowed_cats
    ]

    total_q = len(age_items)

    if not total_q:
        st.warning("Brak pytań dla wybranej grupy wiekowej.")
        st.stop()

    # --- DZIENNA ROTACJA PYTAŃ (5 / dzień) -------------------------
    day_idx = days_since_epoch()
    k_daily = min(5, total_q)

    daily_items = pick_daily_chunk(
        age_items,
        k_daily,
        day_idx,
        f"image_quiz_{age_label}",
    )

    st.caption(
        f"Dzisiejszy zestaw: {len(daily_items)} pytań "
        f"(z {total_q} dostępnych dla {age_label})."
    )

    # --- WYŚWIETLANIE PYTAŃ ---------------------------------------
    for i, t in enumerate(daily_items, start=1):
        img_file = t.get("image")
        img_path = os.path.join(DATA_DIR, "quiz_images", img_file)

        try:
            st.image(img_path, use_container_width=True)
        except Exception:
            st.caption(f"(Nie udało się wczytać obrazu: {img_path})")

        q = t.get("q", "")
        opts = t.get("options", [])
        corr = int(t.get("correct", 0))
        cat = (t.get("category") or "inne")

        st.markdown(f"**{i}. {q}**")

        key_base = f"iq_flat_{age_label}_{i}"
        choice = st.radio(
            "Wybierz:",
            opts,
            key=key_base,
            label_visibility="collapsed",
            index=None,
        )

        if st.button("Sprawdź ✅", key=f"{key_base}_check"):
            if choice is None:
                st.warning("Wybierz odpowiedź.")
            else:
                # przygotowanie stabilnego ID pytania
                try:
                    short_q = q if len(q) <= 60 else q[:57] + "..."
                    base = f"iq::{cat}::{q}"
                    qid = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
                except Exception:
                    short_q = q[:60]
                    qid = None

                if opts and opts.index(choice) == corr:
                    st.success("✅ Dobrze!")
                    st.session_state.xp += 2
                    st.session_state.stickers.add("sticker_hawkeye")

                    # log poprawnej odpowiedzi w quizie obrazkowym
                    try:
                        log_event(
                            f"quiz_ok::image::{cat}::{qid or ''}::{short_q}"
                        )
                    except Exception:
                        pass
                else:
                    if opts:
                        st.error(f"❌ Nie. Poprawna: **{opts[corr]}**.")
                    else:
                        st.error("Brak opcji odpowiedzi w danych quizu.")

                    # log błędnej odpowiedzi z informacją o pomyłce
                    try:
                        chosen = choice or ""
                        correct_label = opts[corr] if opts else ""
                        log_event(
                            f"quiz_fail::image::{cat}::{qid or ''}::{short_q}::{chosen}::{correct_label}"
                        )
                    except Exception:
                        pass



elif page == "Album naklejek":
    st.markdown(f"<div class='big-title'>🏷️ {KID_EMOJI} Album naklejek</div>", unsafe_allow_html=True)
    stickers = list(st.session_state.get("stickers", []))
    if not stickers:
        st.caption("Brak naklejek — zdobywaj je, odpowiadając poprawnie!")
    else:
        for s in stickers:
            meta = STICKERS.get(s, {"emoji":"🏷️","label":s})
            st.markdown(f"- {meta['emoji']} **{meta.get('label', s)}**")

elif page == "Pomoce szkolne":
    st.markdown(f"<div class='big-title'>🧭 {KID_EMOJI} Pomoce szkolne</div>", unsafe_allow_html=True)
    st.caption("Streszczenia lektur i przygotowanie do karty rowerowej.")

    tab_lektury, tab_rower = st.tabs(["Streszczenia lektur", "Moja karta rowerowa"])

    # --- Streszczenia lektur ---
    with tab_lektury:
        lektury_path = os.path.join(DATA_DIR, "lektury.json")
        lektury_db = safe_load_json(lektury_path, default={})
        if not lektury_db:
            st.info("Uzupełnij plik data/lektury.json, aby korzystać z modułu lektur.")
        else:
            groups = sorted(lektury_db.keys())
            group = st.selectbox("Wybierz grupę wiekową:", groups)
            books = lektury_db.get(group, [])
            if not books:
                st.warning("Brak lektur dla tej grupy.")
            else:
                labels = [
                    f"{b.get('title','Bez tytułu')} — {b.get('author','?')}"
                    for b in books
                ]
                idx_book = st.selectbox(
                    "Wybierz lekturę:",
                    options=list(range(len(books))),
                    format_func=lambda i: labels[i],
                )
                book = books[idx_book]
                st.markdown(f"### {book.get('title','Bez tytułu')}")
                st.caption(f"Autor: **{book.get('author','?')}**")

# --- Progres lektur powiązany z kontem dziecka ---
                book_id = book.get("id")
                user = st.session_state.get("user")

                if user and book_id:
                    profile = _user_db_get(user) or {}
                    read_list = profile.get("lektury_read", [])
                    already_read = book_id in read_list

                    if already_read:
                        st.success("✅ Ta lektura jest już oznaczona jako zaliczona.")
                    else:
                        if st.button(
                            "✔️ Oznacz jako przeczytaną / powtórzoną",
                            key=f"lektura_read_btn_{book_id}",
                        ):
                            read_list = list(read_list)
                            if book_id not in read_list:
                                read_list.append(book_id)
                            profile["lektury_read"] = read_list
                            _user_db_set(user, profile)
                            save_progress()

                            st.success("Lektura oznaczona jako przeczytana/powtórzona. 📚")
                            st.experimental_rerun()



                st.markdown("#### Streszczenie")
                summary = book.get("summary_long") or book.get("summary_short") or "Brak streszczenia."
                st.write(summary)

                col1, col2 = st.columns(2)
                with col1:
                    chars = book.get("characters") or []
                    if chars:
                        st.markdown("#### Bohaterowie")
                        for ch in chars:
                            st.markdown(f"- {ch}")

                    themes = book.get("themes") or []
                    if themes:
                        st.markdown("#### Motywy i tematy")
                        for t in themes:
                            st.markdown(f"- {t}")

                with col2:
                    questions = book.get("questions") or []
                    if questions:
                        st.markdown("#### Pytania do przemyślenia")
                        for q in questions:
                            st.markdown(f"- {q}")

                    facts = book.get("facts") or []
                    if facts:
                        st.markdown("#### Ciekawostki")
                        for f in facts:
                            st.markdown(f"- {f}")

                quotes = book.get("quotes") or []
                if quotes:
                    st.markdown("#### Ważne cytaty")
                    for qt in quotes:
                        st.markdown(f"> {qt}")

                plan = book.get("plan") or []
                if plan:
                    st.markdown("#### Plan wydarzeń")
                    for i, step in enumerate(plan, start=1):
                        st.markdown(f"{i}. {step}")

                    # --- Plan odpowiedzi ustnej (5 kroków) ---
                    if st.button(
                        "🎤 Wygeneruj plan odpowiedzi ustnej (5 kroków)",
                        key=f"lektura_plan_ustny_{book_id}",
                    ):
                        st.markdown("#### Pomysł na odpowiedź ustną")
                        core_steps = plan[:5] if len(plan) > 5 else plan
                        for i, step in enumerate(core_steps, start=1):
                            st.markdown(f"{i}. {step}")
                        st.info("Spróbuj opowiedzieć własnymi słowami każdy z punktów – jak przy odpowiedzi przy tablicy.")


 # --- Szybki quiz: 3 pytania ---
                all_q = book.get("questions") or []
                if all_q:
                    st.markdown("### ❓ Szybki quiz – 3 pytania")

                    # Stały dobór pytań dla danej lektury (deterministyczny, żeby dzieci miały powtarzalny zestaw)
                    if len(all_q) <= 3:
                        quiz_qs = all_q
                    else:
                        rnd = random.Random(f"{book_id}_quiz")
                        quiz_qs = rnd.sample(all_q, 3)

                    for i, q in enumerate(quiz_qs, start=1):
                        st.markdown(f"**Pytanie {i}:** {q}")
                        st.text_input(
                            "Twoja odpowiedź:",
                            key=f"lektura_quiz_{book_id}_{i}",
                            placeholder="Napisz własnymi słowami...",
                        )

                    st.caption("To nie jest test na ocenę – po prostu spróbuj odpowiedzieć własnymi słowami 🙂")

                    # --- XP ZA QUIZ, NIE ZA SAMO KLIKNIĘCIE ---
                    user = st.session_state.get("user")
                    if user and book_id:
                        profile = _user_db_get(user) or {}
                        read_list = profile.get("lektury_read", [])
                        already_done = book_id in read_list

                        if already_done:
                            st.success("✅ Lektura zaliczona – XP już przyznane.")
                        else:
                            if st.button(
                                "🎉 Zaliczone! Przyznaj XP za tę lekturę",
                                key=f"lektura_quiz_xp_{book_id}",
                            ):
                                # Zaznaczamy lekturę jako zaliczoną
                                read_list = list(read_list)
                                if book_id not in read_list:
                                    read_list.append(book_id)
                                profile["lektury_read"] = read_list
                                _user_db_set(user, profile)

                                # XP dopiero po quzie
                                st.session_state.xp += 4
                                save_progress()

                                st.success("Brawo! +4 XP za pracę z tą lekturą. 📚🚀")
                                st.experimental_rerun()



    # --- Moja karta rowerowa ---
    with tab_rower:
        st.markdown("### 🚴 Moja karta rowerowa")
        teoria_path = os.path.join(DATA_DIR, "rower", "rower_teoria.json")
        znaki_path = os.path.join(DATA_DIR, "rower", "rower_znaki.json")
        quiz_path = os.path.join(DATA_DIR, "rower", "rower_quiz.json")

        teoria = safe_load_json(teoria_path, default={})
        znaki = safe_load_json(znaki_path, default={})
        quiz = safe_load_json(quiz_path, default={})

        if not teoria and not znaki and not quiz:
            st.info("Dodaj pliki data/rower/rower_teoria.json, rower_znaki.json i rower_quiz.json, aby korzystać z modułu karty rowerowej.")
        else:
            # --- Pasek postępu przygotowań ---
            user = st.session_state.get("user")

            sections = teoria.get("sections", []) if isinstance(teoria, dict) else []
            total_topics = sum(len(sec.get("topics", [])) for sec in sections)
            total_questions = len(quiz.get("questions", [])) if isinstance(quiz, dict) else 0

            viewed_topics = 0
            quiz_correct_sum = 0
            hard_count = 0

            if user:
                profile = _user_db_get(user) or {}
                rower_data = profile.get("rower", {})
                viewed_topics = len(rower_data.get("theory_viewed", []))
                quiz_correct_sum = int(rower_data.get("quiz_correct", 0))
                hard_map = rower_data.get("hard_questions", {})
                hard_count = sum(1 for _qid, cnt in hard_map.items() if cnt >= 2)

            theory_progress = (viewed_topics / total_topics) if total_topics else 0.0
            # zakładamy, że docelowo dobrze odpowiesz przynajmniej raz na każde pytanie
            quiz_progress = min(1.0, quiz_correct_sum / total_questions) if total_questions else 0.0

            overall = 0.5 * theory_progress + 0.5 * quiz_progress

            st.progress(
                overall,
                text=f"Postęp przygotowań: {int(overall*100)}% (teoria + quiz)"
            )

            st.caption(
                f"Teoria: {viewed_topics}/{total_topics} tematów • "
                f"Quiz: {quiz_correct_sum} trafionych odpowiedzi "
                f"(docelowo {total_questions})."
            )
            if hard_count:
                st.caption(f"Masz {hard_count} pytania(a), które sprawiają Ci kłopot – zobacz sekcję „Moje najtrudniejsze pytania” w quizie.")

            sub_teoria, sub_znaki, sub_quiz = st.tabs(["Teoria", "Znaki", "Quiz"])

            # ---------- TEORIA ----------
            with sub_teoria:
                sections = teoria.get("sections", [])
                if not sections:
                    st.info("Brak sekcji teorii w pliku.")
                else:
                    section_ids = [s.get("id", f"sec_{i}") for i, s in enumerate(sections)]
                    section_labels = {
                        s_id: sections[i].get("label", sections[i].get("id", s_id))
                        for i, s_id in enumerate(section_ids)
                    }
                    sec_choice = st.selectbox(
                        "Wybierz dział:",
                        options=section_ids,
                        format_func=lambda sid: section_labels.get(sid, sid),
                    )
                    sec_idx = section_ids.index(sec_choice)
                    sec = sections[sec_idx]
                    topics = sec.get("topics", [])
                    if not topics:
                        st.info("Brak tematów w tym dziale.")
                    else:
                        topic_ids = [t.get("id", f"t_{i}") for i, t in enumerate(topics)]
                        topic_labels = {
                            t_id: topics[i].get("title", topics[i].get("id", t_id))
                            for i, t_id in enumerate(topic_ids)
                        }
                        topic_choice = st.selectbox(
                            "Wybierz temat:",
                            options=topic_ids,
                            format_func=lambda tid: topic_labels.get(tid, tid),
                        )
                        t_idx = topic_ids.index(topic_choice)
                        topic = topics[t_idx]

                        # --- oznaczenie obejrzanego tematu ---
                        user = st.session_state.get("user")
                        if user:
                            profile = _user_db_get(user) or {}
                            rower_data = profile.setdefault("rower", {})
                            viewed = set(rower_data.get("theory_viewed", []))
                            topic_key = f"{sec_choice}:{topic_choice}"
                            if topic_key not in viewed:
                                viewed.add(topic_key)
                                rower_data["theory_viewed"] = list(viewed)
                                profile["rower"] = rower_data
                                _user_db_set(user, profile)

                        st.markdown(f"#### {topic.get('title','Temat')}")
                        st.write(topic.get("text", ""))

                        bullets = topic.get("bullet_points") or []
                        if bullets:
                            st.markdown("**Najważniejsze punkty:**")
                            for b in bullets:
                                st.markdown(f"- {b}")

                        tip = topic.get("tip")
                        if tip:
                            st.info(tip)

            # ---------- ZNAKI ----------
            with sub_znaki:
                categories = znaki.get("categories", [])
                if not categories:
                    st.info("Brak znaków w pliku.")
                else:
                    cat_ids = [c.get("id", f"cat_{i}") for i, c in enumerate(categories)]
                    cat_labels = {
                        c_id: categories[i].get("label", categories[i].get("id", c_id))
                        for i, c_id in enumerate(cat_ids)
                    }
                    cat_choice = st.selectbox(
                        "Wybierz kategorię znaków:",
                        options=cat_ids,
                        format_func=lambda cid: cat_labels.get(cid, cid),
                    )
                    c_idx = cat_ids.index(cat_choice)
                    cat = categories[c_idx]

                    for sign in cat.get("signs", []):
                        header = f"{sign.get('code','?')} — {sign.get('name','(bez nazwy)')}"
                        with st.expander(header):
                            code = sign.get("code", "").replace("/", "_")
                            img_file = os.path.join("rower_signs", f"{code}.png")

                            if os.path.exists(img_file):
                                st.image(img_file, width=140)
                            else:
                                st.caption(f"(Brak obrazka: {img_file})")

                            st.markdown(f"**Opis:** {sign.get('description','')}")
                            st.markdown(f"**Przykład:** {sign.get('example','')}")

            # ---------- QUIZ ----------
            with sub_quiz:
                items = quiz.get("questions", [])
                if not items:
                    st.info("Brak pytań w pliku quizu.")
                else:
                    total_items = len(items)

                    # ile pytań na zestaw (dla Nauki i Egzaminu)
                    k_batch = min(10, total_items)

                    mode = st.radio(
                        "Tryb pracy:",
                        ["Nauka", "Egzamin próbny"],
                        horizontal=True,
                        key="rower_quiz_mode",
                    )


                    # === TRYB NAUKA ===
                    if mode == "Nauka":
                        # numer „zestawu nauki” w tej sesji – żeby dało się wylosować nowe
                        learn_batch = st.session_state.get("rower_learn_batch", 0)

                        # losujemy k_batch pytań bez powtórzeń w ramach zestawu
                        if total_items <= k_batch:
                            learn_items = items
                        else:
                            rnd = random.Random(f"rower_learn_{learn_batch}_{total_items}")
                            learn_items = rnd.sample(items, k_batch)

                        st.caption(
                            f"Zestaw nauki #{learn_batch + 1}: {len(learn_items)} pytań "
                            f"(z {total_items} w całej bazie)."
                        )

                        for i, q in enumerate(learn_items, start=1):
                            st.markdown(f"**{i}. {q.get('question','')}**")
                            options = q.get("options", [])
                            if not options:
                                continue
                            correct_idx = int(q.get("correct", 0))
                            choice = st.radio(
                                "Wybierz odpowiedź:",
                                options,
                                key=f"rower_q_{learn_batch}_{i}",
                                label_visibility="collapsed",
                                index=None,
                            )
                            if st.button("Sprawdź", key=f"rower_q_check_{learn_batch}_{i}"):
                                if choice is None:
                                    st.warning("Najpierw wybierz odpowiedź.")
                                else:
                                    user = st.session_state.get("user")
                                    if options.index(choice) == correct_idx:
                                        st.success("✅ Dobrze!")
                                    else:
                                        st.error(
                                            f"❌ Nie, prawidłowa odpowiedź to: "
                                            f"**{options[correct_idx]}**."
                                        )
                                        # zapamiętujemy trudne pytanie
                                        if user:
                                            qid = q.get("id")
                                            if qid:
                                                profile = _user_db_get(user) or {}
                                                rower_data = profile.setdefault("rower", {})
                                                hard = rower_data.get("hard_questions", {})
                                                hard[qid] = int(hard.get(qid, 0)) + 1
                                                rower_data["hard_questions"] = hard
                                                profile["rower"] = rower_data
                                                _user_db_set(user, profile)
                                    expl = q.get("explanation")
                                    if expl:
                                        st.info(expl)

                        # przycisk: nowy zestaw nauki
                        if st.button("🔁 Wylosuj nowy zestaw pytań do nauki"):
                            st.session_state["rower_learn_batch"] = learn_batch + 1
                            st.experimental_rerun()


                    # === TRYB EGZAMIN PRÓBNY ===
                    else:
                        today_key = get_today_key()

                        # Jeśli weszliśmy w nowy dzień – resetujemy egzamin.
                        if st.session_state.get("rower_exam_date") != today_key:
                            st.session_state["rower_exam_initialized"] = False

                        if not st.session_state.get("rower_exam_initialized", False):
                            # NOWY egzamin = NOWY losowy zestaw pytań
                            st.session_state["rower_exam_initialized"] = True
                            st.session_state["rower_exam_date"] = today_key
                            st.session_state["rower_exam_current"] = 0
                            st.session_state["rower_exam_correct"] = 0
                            st.session_state["rower_exam_recorded"] = False

                            if total_items <= k_batch:
                                exam_items = items
                            else:
                                # losowo bez powtórzeń
                                rnd = random.Random()  # systemowy seed
                                exam_items = rnd.sample(items, k_batch)

                            st.session_state["rower_exam_items"] = exam_items

                        exam_items = st.session_state["rower_exam_items"]
                        cur = st.session_state["rower_exam_current"]

                        st.caption(
                            f"Egzamin próbny: bieżący zestaw to {len(exam_items)} pytań "
                            f"(z {total_items} w całej bazie)."
                        )

                        exam_items = st.session_state["rower_exam_items"]
                        cur = st.session_state["rower_exam_current"]

                        # Koniec egzaminu
                        if cur >= len(exam_items):
                            total = len(exam_items)
                            correct = st.session_state["rower_exam_correct"]
                            percent = int(round(correct * 100 / total)) if total else 0

                            st.success(
                                f"Twój wynik: {correct} / {total} poprawnych odpowiedzi ({percent}%)."
                            )

                            user = st.session_state.get("user")
                            if user and total > 0 and not st.session_state.get("rower_exam_recorded", False):
                                profile = _user_db_get(user) or {}
                                rower_data = profile.setdefault("rower", {})
                                rower_data["quiz_total"] = int(rower_data.get("quiz_total", 0)) + total
                                rower_data["quiz_correct"] = int(rower_data.get("quiz_correct", 0)) + correct
                                best = int(rower_data.get("exam_best_score", 0))
                                if percent > best:
                                    rower_data["exam_best_score"] = percent
                                profile["rower"] = rower_data
                                _user_db_set(user, profile)
                                st.session_state["rower_exam_recorded"] = True

                            passed = percent >= 80 and total >= 5

                            if passed:
                                st.success("Egzamin próbny zaliczony – świetna robota! 🎉")
                                if user:
                                    today_str = today_key
                                    pdf_bytes = generate_rower_certificate_pdf(
                                        username=user,
                                        date_str=today_str,
                                        correct=correct,
                                        total=total,
                                        percent=percent,
                                    )
                                    st.download_button(
                                        "📄 Pobierz certyfikat treningu (PDF)",
                                        data=pdf_bytes,
                                        file_name=f"certyfikat_karta_rowerowa_{today_key}.pdf",
                                        mime="application/pdf",
                                    )
                            else:
                                st.info("Brakuje jeszcze trochę do zaliczenia egzaminu próbnego. Poćwicz i spróbuj ponownie 💪")

                            if st.button("Rozpocznij nowy egzamin"):
                                st.session_state["rower_exam_initialized"] = False
                                st.session_state["rower_exam_recorded"] = False
                                st.rerun()
                            st.stop()


                        # Bieżące pytanie
                        q = exam_items[cur]
                        st.markdown(f"**Pytanie {cur + 1} z {len(exam_items)}**")
                        st.markdown(q.get("question", ""))

                        options = q.get("options", [])
                        if not options:
                            st.warning("Brak odpowiedzi dla tego pytania.")
                            st.stop()

                        correct_idx = int(q.get("correct", 0))
                        choice = st.radio(
                            "Wybierz odpowiedź:",
                            options,
                            key=f"rower_exam_q_{cur}",
                            label_visibility="collapsed",
                            index=None,
                        )

                        if st.button("Zatwierdź odpowiedź", key=f"rower_exam_check_{cur}"):
                            if choice is None:
                                st.warning("Najpierw wybierz odpowiedź.")
                            else:
                                if options.index(choice) == correct_idx:
                                    st.success("✅ Dobrze!")
                                    st.session_state["rower_exam_correct"] += 1
                                else:
                                    st.error(
                                        f"❌ Nie, prawidłowa odpowiedź to: "
                                        f"**{options[correct_idx]}**."
                                    )
                                expl = q.get("explanation")
                                if expl:
                                    st.info(expl)
                                st.session_state["rower_exam_current"] += 1
                                st.rerun()

                    # --- Moje najtrudniejsze pytania (na podstawie historii błędów) ---
                    user = st.session_state.get("user")
                    if user:
                        profile = _user_db_get(user) or {}
                        rower_data = profile.get("rower", {})
                        hard_map = rower_data.get("hard_questions", {})

                        if hard_map:
                            full_questions = {q.get("id"): q for q in items}
                            hardest = sorted(
                                hard_map.items(),
                                key=lambda kv: kv[1],
                                reverse=True,
                            )
                            display_list = []
                            for qid, cnt in hardest:
                                q_obj = full_questions.get(qid)
                                if q_obj:
                                    display_list.append((q_obj.get("question", ""), cnt))

                            if display_list:
                                with st.expander("😬 Moje najtrudniejsze pytania"):
                                    for text, cnt in display_list[:5]:
                                        st.markdown(f"- **{text}** — pomyłka {cnt}×")
                        else:
                            st.caption("Na razie brak „trudnych pytań” – dopiero zbieramy dane z quizów 🙂")
                    else:
                        st.caption("Zaloguj się, aby śledzić swoje trudne pytania i postęp przygotowań.")

elif page == "Przedmioty szkolne":
    st.markdown(f"<div class='big-title'>📚 {KID_EMOJI} Przedmioty szkolne</div>", unsafe_allow_html=True)
    st.caption("Codziennie 10 pytań MCQ na przedmiot i grupę wiekową.")

    # Helpers
    import hashlib, random, math
    from datetime import date, datetime

    def _mcq_key(subj: str, idx: int):
        return f"mcq_{subj}_{idx}"

    def _stable_shuffle(arr, seed_text: str):
        arr = list(arr)
        rnd = random.Random(int(hashlib.sha256(seed_text.encode('utf-8')).hexdigest(), 16) % (10**12))
        rnd.shuffle(arr)
        return arr

    def pick_daily_chunk(items, k, day_idx: int, salt: str):
        if not items:
            return []
        k = max(1, min(k, len(items)))
        shuffled = _stable_shuffle(items, salt)
        num_chunks = math.ceil(len(shuffled) / k)
        start = (day_idx % num_chunks) * k
        return shuffled[start:start+k]

    # Load tasks
    try:
        TASKS = load_tasks()
    except Exception:
        try:
            import json
            TASKS = json.load(open("data/tasks.json","r",encoding="utf-8"))
        except Exception:
            try:
                TASKS = json.load(open("tasks.json","r",encoding="utf-8"))
            except Exception:
                TASKS = {}

    today_str = datetime.now().strftime("%Y-%m-%d")
    day_idx = (date.today() - date(2025,1,1)).days
    age_group = st.session_state.get("age_group", "10-12")
    subject_defs = [
        ("matematyka", "Matematyka"),
        ("polski", "Język polski"),
        ("historia", "Historia"),
        ("geografia", "Geografia"),
        ("fizyka", "Fizyka"),
        ("chemia", "Chemia"),
        ("angielski", "Angielski"),
        ("niemiecki", "Niemiecki"),
        ("biologia", "Biologia"),
        ("informatyka", "Informatyka"),
        ("wos", "WOS"),
        ("muzyka", "Muzyka"),
        ("religie_swiata", "Religie świata i tradycje"),
        ("etyka", "Etyka i wartości"),
        ("wf", "WF - zdrowie i sport"),
        ("logika", "Logika & problem solving"),
    ]

    subjects = ["matematyka","polski","historia","geografia","fizyka","chemia","angielski","niemiecki","biologia", "informatyka", "wos", "muzyka", "religie_swiata", "etyka", "wf", "logika",]

    def tasks_for(subject: str, group: str):
        subj = TASKS.get(subject, {})
        arr = subj.get(group, [])
        return arr if isinstance(arr, list) else []

    if st.session_state.get("daily_subject_tasks_date") != today_str or "daily_subject_tasks" not in st.session_state:
        st.session_state.daily_subject_tasks_date = today_str
        st.session_state.daily_subject_tasks = {}
        for s in subjects:
            pool = tasks_for(s, age_group)
            chosen = pick_daily_chunk(pool, 10, day_idx, f"{s}:{age_group}:{today_str}")
            st.session_state.daily_subject_tasks[s] = chosen

    st.info(f"Dzisiejsza data: {today_str} | Grupa: {age_group}")

    def show_subject(subj_key: str, title: str):
        items = st.session_state.daily_subject_tasks.get(subj_key, [])
        st.markdown(f"#### {title} · Dzisiejsze pytania ({len(items)})")

        if not items:
            st.caption("Brak pytań dla tej grupy. Uzupełnij tasks.json.")
            return

        for i, t in enumerate(items, start=1):
            if not (isinstance(t, dict) and t.get("type") == "mcq"):
                st.error(f"Pozycja #{i} nie jest MCQ. Sprawdź tasks.json.")
                continue

            q = t.get("q", f"Pytanie {i}")
            opts = list(t.get("options", []))
            corr = int(t.get("correct", 0))
            base = _mcq_key(subj_key, i)

            st.markdown(f"**{i}. {q}**")
            choice = st.radio("Wybierz odpowiedź:", options=opts, index=None, key=base+"_choice", label_visibility="collapsed")
            if st.button("Sprawdź ✅", key=base+"_check"):
                if choice is None:
                    st.warning("Wybierz odpowiedź.")
                else:
                    ok = (opts.index(choice) == corr)
                    if ok:
                        st.success("✅ Dobrze! +5 XP")
                        try:
                            u = st.session_state.get("user") or "(anon)"
                            mark_task_done(u, subj_key, q, xp_gain=5)
                        except Exception:
                            pass
                    else:
                        st.error(f"❌ Niepoprawnie. Prawidłowa odpowiedź: **{opts[corr]}**.")

 # Zamiast zakładek: jedno rozwijane menu
    subject_labels = [label for _, label in subject_defs]

    selected_label = st.selectbox(
        "Wybierz przedmiot",
        subject_labels,
        key="school_subject_select",
    )

    # znajdź klucz po etykiecie
    selected_key = next(
        key for key, label in subject_defs if label == selected_label
    )

    show_subject(selected_key, selected_label)


elif page == "Słowniczek":
    st.markdown("# 📖 Słowniczek pojęć")
    st.caption("Hasła są pogrupowane. Możesz też skorzystać z wyszukiwarki.")

    query = st.text_input("Szukaj pojęcia…", "").strip().lower()

    if query:
        # Wyszukiwanie we wszystkich kategoriach
        results = []
        for cat, entries in CATEGORIZED_GLOSSARY.items():
            for k, v in entries.items():
                if query in k.lower():
                    results.append((cat, k, v))
        if not results:
            st.caption("Brak wyników — spróbuj innego słowa.")
        else:
            st.subheader("🔍 Wyniki wyszukiwania")
            for i, (cat, k, v) in enumerate(sorted(results), start=1):
                cols = st.columns([3,1])
                with cols[0]:
                    st.markdown(
    f"**{k}** — {v}  \n<span class='pill'>{cat}</span>",
    unsafe_allow_html=True
)

                with cols[1]:
                    if cat == "ANGIELSKI":
                        tts_button_en(k, key=f"s_{i}")
    else:
        # Przeglądanie kategorii – jedna wybrana z listy rozwijanej
        categories = list(CATEGORIZED_GLOSSARY.keys())
        if not categories:
            st.info("Słowniczek jest jeszcze pusty. Dodaj pliki w folderze data/glossary.")
            st.stop()

        selected_cat = st.selectbox(
            "Wybierz przedmiot:",
            categories,
            index=0,
        )

        entries = CATEGORIZED_GLOSSARY.get(selected_cat, {})

        for i, (k, v) in enumerate(sorted(entries.items()), start=1):
            cols = st.columns([3, 1])
            with cols[0]:
                st.write(f"**{k}** — {v}")
            with cols[1]:
                if selected_cat == "ANGIELSKI":
                    tts_button_en(k, key=f"{selected_cat}_{i}")

elif page == "Moje osiągnięcia":
    st.markdown("# ⭐ Moje osiągnięcia")
    st.caption("Tutaj zobaczysz swój poziom, XP, odznaki i naklejki.")

    # Główne liczby – podobnie jak w Panelu rodzica
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Poziom", current_level(st.session_state.xp))
    col2.metric("XP", st.session_state.xp)
    col3.metric("Odznaki", len(st.session_state.badges))
    col4.metric("Naklejki", len(st.session_state.stickers))

    st.divider()

    # Imię + grupa wiekowa
    st.subheader("🧒 Twój profil")
    st.write(f"Imię/nick: **{st.session_state.kid_name or '(bez imienia)'}**")
    st.write(f"Wiek: **{st.session_state.age}** lat")
    st.write(f"Grupa wiekowa: **{st.session_state.age_group}**")

    st.divider()

    # Odznaki
    st.subheader("🏅 Odznaki")
    if st.session_state.badges:
        for b in sorted(list(st.session_state.badges)):
            st.markdown(f"- {b}")
    else:
        st.caption("Nie masz jeszcze odznak. Sprawdź zakładkę **Misje** i **Quizy**, żeby je zdobyć!")

    st.divider()

    # Naklejki
    st.subheader("📘 Naklejki")
    if st.session_state.stickers:
        st.write(", ".join(sorted(list(st.session_state.stickers))))
        st.caption("Więcej naklejek zdobywasz za misje, quizy i inne zadania.")
    else:
        st.caption("Album jest jeszcze pusty. Zajrzyj do zakładki **Album naklejek** i zacznij kolekcję!")

    st.divider()

    st.subheader("📊 Postępy w Data4Kids")
    st.markdown(
        """
        - Robiąc misje i quizy zdobywasz **XP** i **odznaki**.  
        - Im więcej zadań, tym wyższy **poziom**.  
        - W zakładce **Hall of Fame** możesz zapisać swój profil do wspólnego pliku mistrzów.
        """
    )

    # === Data science: analiza zadań szkolnych dla tego użytkownika ===
    user = st.session_state.get("user")

    if user:
        profile = _user_db_get(user) or {}
        school_tasks = profile.get("school_tasks", {})

        if school_tasks:
            import pandas as pd

            rows = []
            for day, subj_map in school_tasks.items():
                for subj, tasks in subj_map.items():
                    rows.append(
                        {
                            "data": day,
                            "przedmiot": subj,
                            "zadania": len(tasks),
                        }
                    )

            if rows:
                df = pd.DataFrame(rows)

                st.subheader("📈 Ile zadań zrobiono z każdego przedmiotu?")
                subj_counts = (
                    df.groupby("przedmiot")["zadania"]
                    .sum()
                    .sort_values(ascending=False)
                )
                st.bar_chart(subj_counts)

                best_subject = subj_counts.idxmax()
                worst_subject = subj_counts.idxmin()

                st.subheader("🤖 Podpowiedź Data4Kids")
                if best_subject == worst_subject:
                    st.write(
                        "Na razie masz zadania tylko z jednego przedmiotu – "
                        "spróbuj dorzucić coś z innego, żeby było ciekawiej. 🙂"
                    )
                else:
                    st.write(
                        f"Najwięcej zadań zrobiłeś z: **{best_subject}** – super! 💪"
                    )
                    st.write(
                        f"Najmniej ćwiczysz: **{worst_subject}** – "
                        "może dziś zrobisz jedno zadanie właśnie z tego przedmiotu? 🎯"
                    )
        else:
            st.caption(
                "Gdy zaczniesz oznaczać zadania jako zrobione w zakładce "
                "**Przedmioty szkolne**, pojawią się tutaj statystyki i podpowiedzi."
            )
    else:
        st.caption("Zaloguj się, żeby zobaczyć swoje statystyki zadań.")

elif page == "Hall of Fame":
    st.markdown("# 🏆 Hall of Fame")
    st.caption("Ranking oparty o dane z kont użytkowników (users.json).")

    # --- 1. Wczytanie bazy profili z users.json ---
    db = _load_users()
    rows = []
    for login, prof in db.items():
        xp = int(prof.get("xp", 0) or 0)
        age_group = prof.get("age_group") or "brak"
        age_val = prof.get("age")
        try:
            age_val = int(age_val) if age_val is not None else None
        except Exception:
            age_val = None

        rows.append(
            {
                "login": login,
                "xp": xp,
                "level": current_level(xp),
                "age_group": age_group,
                "age": age_val,
            }
        )

    if not rows:
        st.info(
            "Brak zapisanych profili w bazie users.json – "
            "najpierw załóż kilka kont na stronie **Start**."
        )
    else:
        df_rank = pd.DataFrame(rows)

        st.subheader("📊 Top gracze Data4Kids")

        # --- 2. Filtrowanie po grupie wiekowej ---
        groups = sorted(
            g for g in df_rank["age_group"].unique()
            if g and g != "brak"
        )
        group_choice = st.selectbox(
            "Pokaż ranking dla grupy wiekowej:",
            ["Wszystkie grupy"] + groups,
        )

        df_view = df_rank.copy()
        if group_choice != "Wszystkie grupy":
            df_view = df_view[df_view["age_group"] == group_choice]

        # --- 3. Sortowanie po XP + wybór liczby graczy ---
        df_view = df_view.sort_values("xp", ascending=False)
        top_n = st.slider(
            "Ilu graczy pokazać w rankingu?",
            min_value=3,
            max_value=30,
            value=10,
        )
        df_top = df_view.head(top_n).reset_index(drop=True)

        st.dataframe(
            df_top[["login", "level", "xp", "age_group", "age"]],
            use_container_width=True,
        )

        # --- 4. Wykres słupkowy XP vs gracz ---
        if not df_top.empty:
            try:
                chart = (
                    alt.Chart(df_top)
                    .mark_bar()
                    .encode(
                        x=alt.X("xp:Q", title="XP"),
                        y=alt.Y("login:N", sort="-x", title="Gracz"),
                        tooltip=["login", "xp", "level", "age_group", "age"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.caption(f"(Nie udało się narysować wykresu: {e})")

        st.markdown("---")

    # --- 5. Mój profil do pobrania (jak mini-CV Data4Kids) ---
    st.subheader("📁 Mój profil do portfolio")

    profile = {
        "name": st.session_state.kid_name or "(bez imienia)",
        "age": st.session_state.age,
        "age_group": st.session_state.age_group,
        "xp": st.session_state.xp,
        "level": current_level(st.session_state.xp),
        "badges": sorted(list(st.session_state.badges)),
        "stickers": sorted(list(st.session_state.stickers)),
        "dataset": st.session_state.dataset_name,
        "timestamp": datetime.now(tz=tz.gettz("Europe/Warsaw")).isoformat(),
        "missions_done": sorted(
            [k for k, v in st.session_state.missions_state.items() if v.get("done")]
        ),
    }

    st.json(profile)
    st.download_button(
        "Pobierz mój profil (JSON)",
        data=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="data4kids_profile.json",
        mime="application/json",
    )


elif page == "Wsparcie & konkursy":
    st.markdown(
        f"<div class='big-title'>💝 {KID_EMOJI} Wsparcie rozwoju & konkursy</div>",
        unsafe_allow_html=True,
    )
    st.caption("Strefa głównie dla rodziców / opiekunów. Dziękujemy za każde wsparcie! 🙏")

    col_left, col_right = st.columns([2, 1])

    # --- LEWA KOLUMNA: informacje o wsparciu + formularz zgłoszeń ---
    with col_left:
        st.markdown("### Jak możesz wesprzeć projekt?")

        st.write(
            """
            Ten projekt powstaje po godzinach, żeby **dzieci mogły uczyć się danych,
            statystyki i przedmiotów szkolnych w formie zabawy**.  

            Wpłaty pomagają w:
            - opłaceniu serwera i domeny,
            - rozwoju nowych modułów i misji,
            - organizowaniu **konkursów z nagrodami fizycznymi** (książki, gry edukacyjne itp.).
            """
        )

        if any([DONATE_BUYCOFFEE_URL, DONATE_PAYPAL_URL, DONATE_BANK_INFO]):
            st.markdown("#### Dane do wpłaty")

            if DONATE_BUYCOFFEE_URL:
                st.markdown(
                    f"- ☕ Szybka wpłata: [BuyCoffee]({DONATE_BUYCOFFEE_URL})"
                )

            if DONATE_PAYPAL_URL:
                st.markdown(
                    f"- 💳 PayPal: [przejdź do płatności]({DONATE_PAYPAL_URL})"
                )

            if DONATE_BANK_INFO:
                st.markdown("**Przelew tradycyjny:**")
                st.code(DONATE_BANK_INFO, language="text")
        else:
            st.info(
                "Adminie: ustaw `D4K_BUYCOFFEE_URL`, `D4K_PAYPAL_URL` i/lub `D4K_BANK_INFO` "
                "w kodzie lub zmiennych środowiskowych, aby tutaj pokazać konkretne dane do wpłat."
            )


        st.markdown("---")
        st.markdown("### Zgłoszenie do konkursu (po dokonaniu wpłaty)")

        st.write(
            """
            Po dokonaniu wpłaty możesz zgłosić się do konkursu.  
            Zgłoszenia trafiają do pliku `data/donors.json`, z którego można później
            wylosować zwycięzców (po weryfikacji wpłat).
            """
        )

        with st.form("donor_form"):
            parent_name = st.text_input("Imię i nazwisko rodzica / opiekuna")
            contact = st.text_input("E-mail do kontaktu (wysyłka nagrody itp.)")
            child_login = st.text_input("Login dziecka w Data4Kids (opcjonalnie)")
            amount = st.text_input("Przybliżona kwota wsparcia (np. 20 zł)", value="")
            note = st.text_area("Uwagi (np. preferencje nagród, rozmiar T-shirtu 😉)", value="")

            consent = st.checkbox(
                "Oświadczam, że dokonałem/dokonałam wpłaty i akceptuję regulamin konkursu.",
                value=False,
            )

            submitted = st.form_submit_button("Zapisz zgłoszenie do konkursu")

            if submitted:
                if not parent_name or not contact or not consent:
                    st.warning("Uzupełnij imię, e-mail oraz zaznacz akceptację regulaminu.")
                else:
                    donors = _load_donors()
                    donors.append(
                        {
                            "parent_name": parent_name,
                            "contact": contact,
                            "child_login": child_login,
                            "amount": amount,
                            "note": note,
                            "timestamp": datetime.now(tz=tz.gettz("Europe/Warsaw")).isoformat(),
                        }
                    )
                    _save_donors(donors)
                    st.success("Zgłoszenie zapisane. Dziękujemy za wsparcie! 💚")

    with col_right:
        st.markdown("### 📈 Statystyki i ranking")

        donors = _load_donors()
        st.metric("Liczba zgłoszeń konkursowych", len(donors))

        st.markdown("#### Mini-ranking XP (przykład konkursu)")

        group_filter = st.selectbox(
            "Grupa wiekowa dla rankingu:",
            ["Wszystkie", "7-9", "10-12", "13-14"],
            index=0,
        )
        selected_group = None if group_filter == "Wszystkie" else group_filter

        lb = get_leaderboard(limit=10, age_group=selected_group)
        if not lb:
            st.caption("Brak danych o graczach dla tej konfiguracji.")
        else:
            df_lb = pd.DataFrame(lb)
            df_lb.rename(
                columns={
                    "user": "Użytkownik",
                    "xp": "XP",
                    "badges": "Odznaki",
                    "stickers": "Naklejki",
                    "age_group": "Grupa wiekowa",
                },
                inplace=True,
            )
            st.dataframe(df_lb, hide_index=True, use_container_width=True)

    # --- PRAWA KOLUMNA: statystyki i ranking ---
    with col_right:
        st.markdown("### 📈 Statystyki i ranking")

        donors = _load_donors()
        st.metric("Liczba zgłoszeń konkursowych", len(donors))

        st.markdown("#### Mini-ranking XP (przykład konkursu)")
        lb = get_leaderboard(limit=10)
        if not lb:
            st.caption("Brak danych o graczach (nikt jeszcze nie ma XP).")
        else:
            df_lb = pd.DataFrame(lb)
            df_lb.rename(
                columns={"user": "Użytkownik", "xp": "XP", "badges": "Odznaki", "stickers": "Naklejki"},
                inplace=True,
            )
            st.dataframe(df_lb, hide_index=True, use_container_width=True)

        st.markdown(
            """
            Możesz np. zorganizować:
            - konkurs „**Top 3 XP w danym miesiącu**”,
            - losowanie nagród **wśród wszystkich zgłoszonych darczyńców**,
            - specjalne naklejki / odznaki za udział w konkursie.
            """
        )


elif page == "Regulamin":
    st.markdown("# 📜 Regulamin Data4Kids")
    st.caption(f"Wersja aplikacji: v{VERSION}")

    # --- Regulamin aplikacji / prywatności ---
    st.markdown("""
1. **Przechowywanie danych.**
   Aplikacja korzysta z bazy danych działającej na serwerze twórcy aplikacji. 
   Dane użytkowników są przechowywane wyłącznie na tym serwerze i nie są przekazywane osobom trzecim ani wykorzystywane do celów komercyjnych. Nie stosujemy zewnętrznej analityki ani śledzenia.
   Dane są wykorzystywane wyłącznie do działania aplikacji (logowanie, profile, posty, statystyki wewnętrzne).

2. **Brak danych osobowych.** Nie prosimy o imię i nazwisko ani e-mail.  
   Login w aplikacji może być **pseudonimem**.

3. **Hasła i bezpieczeństwo.** Hasła są haszowane (z solą) i zapisywane lokalnie.  
   Dbaj o silne hasło i nie udostępniaj go innym.

4. **Profil dziecka.** Postępy (XP, odznaki, naklejki) zapisywane są **lokalnie** w pliku `data/users.json`.  
   Możesz je w każdej chwili usunąć w **Panelu rodzica**.

5. **PIN rodzica.** Panel rodzica jest zabezpieczony PIN-em ustawianym lokalnie w aplikacji.

6. **Treści edukacyjne.** Aplikacja ma charakter edukacyjny i **nie zastępuje** zajęć szkolnych.  
   Dokładamy starań, by treści były poprawne, ale mogą się zdarzyć błędy.

7. **Pliki użytkownika.** Jeżeli wgrywasz własne dane (np. CSV), pozostają one na Twoim urządzeniu.

8. **Odpowiedzialne korzystanie.** Korzystaj z aplikacji zgodnie z prawem i zasadami dobrego wychowania.

9. **Zmiany regulaminu.** Regulamin może się zmienić wraz z rozwojem aplikacji; aktualna wersja jest zawsze tutaj.
    """)

    st.divider()
    st.subheader("Twoje prawa i opcje")
    st.markdown("""
- **Podgląd danych**: w Panelu rodzica masz wgląd w ostatnie aktywności i ustawienia.  
- **Usuwanie danych**: w Panelu rodzica znajdziesz przyciski do usunięcia **Twojego profilu**.  
- **Brak zgody?** Nie korzystaj z aplikacji i usuń lokalne pliki w katalogu `data/`.
    """)

    st.divider()
    st.subheader("Akceptacja regulaminu")

    accepted_ver = st.session_state.get("accepted_terms_version")

    if accepted_ver == VERSION:
        st.success("Dla tej wersji aplikacji regulamin został już zaakceptowany na tym urządzeniu.")
    else:
        st.info(
            "Przeczytaj regulamin powyżej. Jeśli się zgadzasz, kliknij przycisk poniżej, "
            "aby móc założyć konto w zakładce **Start**."
        )
        if st.button("Przeczytałem/przeczytałam regulamin i akceptuję go"):
            st.session_state["accepted_terms_version"] = VERSION
            st.success("Dziękujemy! Możesz teraz założyć konto w zakładce Start.")


    # --- Regulamin konkursu ---
    st.markdown(
        "<div class='big-title'>📜 Regulamin konkursu Data4Kids</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
## 1. Postanowienia ogólne
1. Niniejszy regulamin określa zasady udziału w konkursach organizowanych w ramach projektu **Data4Kids** (dalej: „Konkurs”).
2. Organizatorem Konkursu jest właściciel i administrator aplikacji Data4Kids (dalej: „Organizator”).
3. Konkurs nie jest grą losową, loterią fantową, zakładem wzajemnym ani żadną inną formą gry wymagającą zgłoszenia do właściwych organów administracyjnych.
4. Konkurs jest przeprowadzany w celach edukacyjnych i promocyjnych, a nagrody mają charakter drobnych upominków rzeczowych.

## 2. Uczestnicy
1. Uczestnikiem Konkursu może być osoba pełnoletnia działająca jako rodzic lub opiekun legalny dziecka korzystającego z aplikacji Data4Kids.
2. Rodzic/opiekun zgłasza udział dziecka w Konkursie poprzez formularz dostępny w zakładce **„Wsparcie & konkursy”**.
3. Zgłoszenie udziału oznacza akceptację niniejszego regulaminu.

## 3. Zasady uczestnictwa
1. Warunkiem przystąpienia do Konkursu jest dokonanie dobrowolnego wsparcia projektu poprzez dowolną wpłatę („darowiznę”) lub spełnienie innych warunków określonych w opisie konkretnej edycji Konkursu.
2. Kwota wsparcia nie wpływa na szanse zwycięstwa, chyba że opis Konkursu stanowi inaczej (np. system losów).
3. Zgłoszenie do Konkursu wymaga podania:
   - imienia i nazwiska rodzica/opiekuna,
   - adresu e-mail do kontaktu,
   - opcjonalnie loginu dziecka w aplikacji.
4. Wszystkie dane są wykorzystywane wyłącznie do przeprowadzenia Konkursu oraz kontaktu z osobami nagrodzonymi.

## 4. Przebieg i rozstrzygnięcie Konkursu
1. Losowanie zwycięzców odbywa się z wykorzystaniem narzędzia dostępnego w panelu administratora aplikacji Data4Kids lub niezależnego skryptu losującego.
2. W zależności od opisu edycji Konkursu losowanie może odbywać się:
   - „każde zgłoszenie = 1 los”,
   - „unikalny adres e-mail = 1 los”,
   - według kryteriów punktowych (np. ranking XP dziecka).
3. Wyniki losowania są zapisywane w formie elektronicznej i przechowywane dla celów dowodowych przez Organizatora.
4. Organizator skontaktuje się ze zwycięzcami drogą e-mailową w celu ustalenia formy przekazania nagrody.

## 5. Nagrody
1. Nagrody mają charakter upominków rzeczowych (np. książki edukacyjne, gry logiczne, zestawy kreatywne).
2. Nagrody nie podlegają wymianie na gotówkę ani inne świadczenia.
3. Organizator pokrywa koszty wysyłki nagród na terenie Polski.
4. W przypadku braku kontaktu ze strony zwycięzcy przez **14 dni** od ogłoszenia wyników, nagroda przepada i może zostać przyznana innej osobie.

## 6. Dane osobowe
1. Administratorem danych osobowych jest Organizator.
2. Dane uczestników są przetwarzane wyłącznie na potrzeby przeprowadzenia Konkursu i przekazania nagród.
3. Uczestnik ma prawo dostępu do swoich danych, ich poprawiania oraz żądania usunięcia.
4. Dane nie są przekazywane podmiotom trzecim.

## 7. Reklamacje
1. Reklamacje dotyczące Konkursu można kierować do Organizatora na adres kontaktowy wskazany w aplikacji.
2. Reklamacje będą rozpatrywane w terminie do 14 dni od ich zgłoszenia.
3. Decyzja Organizatora w sprawie reklamacji jest ostateczna.

## 8. Postanowienia końcowe
1. Organizator zastrzega sobie prawo do zmian regulaminu, o ile nie wpływają one na prawa uczestników zdobyte przed zmianą.
2. Organizator może unieważnić Konkurs w przypadku stwierdzenia nadużyć lub zdarzeń losowych uniemożliwiających jego prawidłowe przeprowadzenie.
3. W sprawach nieuregulowanych regulaminem zastosowanie mają przepisy prawa polskiego.
    """)

elif page == "Kontakt":
    st.markdown(
        "<div class='big-title'>📮 Kontakt</div>",
        unsafe_allow_html=True,
    )

    st.write(
        """
        Ta zakładka jest przeznaczona dla **rodziców, nauczycieli i opiekunów**, którzy chcą
        skontaktować się w sprawie aplikacji *Data4Kids*.

        Możesz napisać w sprawach:
        - pytań dotyczących działania aplikacji,
        - pomysłów na nowe funkcje,
        - zgłoszeń błędów,
        - współpracy ze szkołą lub zajęciami edukacyjnymi.
        """
    )

    contact_email = "data4kids@proton.me"

    st.subheader("📧 Adres e-mail")
    st.markdown(
        f"**{contact_email}**  \n"
        f"Kliknij tutaj, aby napisać: [mailto:{contact_email}](mailto:{contact_email})"
    )

    st.write("---")
    st.subheader("💬 Formularz kontaktowy")

    st.info("Wypełnij poniższy formularz — to najszybszy sposób kontaktu z zespołem Data4Kids.")

    with st.form("contact_form"):
        name = st.text_input("Imię i nazwisko / szkoła (opcjonalnie)")
        reply_to = st.text_input("E-mail do odpowiedzi")
        topic = st.text_input("Temat wiadomości")
        message = st.text_area("Treść wiadomości")

        sent = st.form_submit_button("Wyślij wiadomość")

    if sent:
        if not reply_to or not message:
            st.warning("Aby wysłać wiadomość, podaj e-mail do kontaktu i treść wiadomości.")
        else:
            first_name = name.split()[0] if name else ""
            st.success(
                f"✅ Dziękujemy za wiadomość{', ' + first_name if first_name else ''}! ✨"
            )
            st.markdown(
                f"""
                Twoja wiadomość trafiła do zespołu **Data4Kids**.  
                Odpowiemy na adres: **{reply_to}**.  

                Jeśli chcesz, możesz też napisać bezpośrednio z poczty na:
                **{contact_email}**.
                """
            )
            st.caption(
                "Uwaga: w tej wersji aplikacji wiadomość nie jest jeszcze wysyłana "
                "automatycznie mailem — to formularz kontaktu z twórcą aplikacji."
            )

# ADMINISTRATOR (TOTP / Authenticator)
# -----------------------------
elif page == "Administrator":
    st.markdown("# 🛡️ Administrator")
    st.caption(
        "Dostęp tylko przez TOTP (Authenticator) — sekret przechowywany bezpiecznie w bazie danych."
    )

    # load/save admin TOTP secret z bazy (helper get_admin_totp_secret)
    secret = get_admin_totp_secret()

    import_base_ok = True
    try:
        import pyotp
        import qrcode
        from PIL import Image  # noqa: F401  (używane przez qrcode/Pillow)
        import io, base64
    except Exception:
        import_base_ok = False

    if not import_base_ok:
        st.error("Brakuje pakietów pyotp/qrcode/pillow. Zainstaluj: pip install pyotp qrcode pillow")
        st.stop()

    # logout button shown if already unlocked
    if st.session_state.get("admin_unlocked", False):
        st.success("Jesteś zalogowany jako Administrator.")
        if st.button("Wyloguj administratora"):
            st.session_state["admin_unlocked"] = False
            st.info("Wylogowano.")
            st.rerun()

    # Jeśli nie ma jeszcze sekretu -> pozwól go utworzyć
    if not secret:
        st.warning(
            "Brak skonfigurowanego TOTP. Utwórz sekret i dodaj go do aplikacji Authenticator na telefonie."
        )
        if st.button("Utwórz sekret TOTP teraz"):
            new_secret = pyotp.random_base32()
            set_admin_totp_secret(new_secret)
            st.success("Sekret wygenerowany. Dodaj go do Authenticator (pokażę QR i secret po zalogowaniu).")
            st.rerun()
        st.stop()

    # Formularz logowania TOTP
    st.markdown("**Zaloguj się kodem z aplikacji Authenticator**")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        code = st.text_input("6-cyfrowy kod TOTP", max_chars=6, key="admin_code_input")
    with col_b:
        if st.button("Zaloguj administratora"):
            try:
                totp = pyotp.TOTP(secret)
                ok = totp.verify(code.strip(), valid_window=1)
                if ok:
                    st.session_state["admin_unlocked"] = True
                    st.success("Zalogowano jako Administrator.")
                    st.rerun()
                else:
                    st.error("Kod niepoprawny. Sprawdź w aplikacji Authenticator i spróbuj ponownie.")
            except Exception as e:
                st.error(f"Błąd weryfikacji: {e}")

    # ⛔ Jeśli nie zalogowany admin – nie pokazujemy sekretu ani panelu
    if not st.session_state.get("admin_unlocked", False):
        st.info("Aby uzyskać dostęp do panelu administratora i konfiguracji sekretu, podaj poprawny kod TOTP.")
        st.stop()

    # ----------------- od tego miejsca UŻYTKOWNIK JEST ADMINEM -----------------
    st.divider()
    st.markdown("### Konfiguracja sekretu (tylko dla zalogowanego administratora)")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            "Jeżeli chcesz skonfigurować ręcznie w aplikacji Authenticator, "
            "użyj poniższego secretu. **Nigdy nie udostępniaj go nikomu innemu.**"
        )
        st.code(secret, language="text")

        if st.button("Wygeneruj nowy sekret TOTP"):
            new_secret = pyotp.random_base32()
            set_admin_totp_secret(new_secret)
            st.success(
                "Wygenerowano nowy sekret. Skonfiguruj go ponownie w aplikacji Authenticator. "
                "Bieżąca sesja administratora została wylogowana."
            )
            st.session_state["admin_unlocked"] = False
            st.rerun()

    with col2:
        if st.button("Pokaż QR (provisioning URI)"):
            try:
                totp = pyotp.TOTP(secret)
                uri = totp.provisioning_uri(
                    name=f"{APP_NAME}-admin",
                    issuer_name=APP_NAME,
                )
                qr = qrcode.make(uri)
                buf = io.BytesIO()
                qr.save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, caption="Zeskanuj ten QR kod w aplikacji Authenticator")
            except Exception as e:
                st.error(f"Nie udało się wygenerować QR: {e}")

    st.markdown("---")

    # if admin unlocked -> show admin controls
    # (tu już wiemy, że admin_unlocked == True)
    st.markdown("## 🔧 Panel administratora — operacje")
    db = _load_users()

    st.subheader("Konta użytkowników")
    if not db or all(k.startswith("_") for k in db.keys()):
        st.caption("Brak użytkowników w bazie danych.")
    else:
        cols = st.columns([2, 1, 1])
        cols[0].markdown("**Login**")
        cols[1].markdown("**XP**")
        cols[2].markdown("**Akcje**")
        # show all real users (exclude internal keys starting with _)
        users_list = [k for k in db.keys() if not k.startswith("_")]
        for u in users_list:
            prof = db.get(u, {})
            xp = prof.get("xp", 0)
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.write(u)
            c2.write(xp)
            if c3.button(f"Usuń konto: {u}", key=f"del_user_{u}"):
                del db[u]
                _save_users(db)
                st.success(f"Usunięto konto: {u}")
                st.rerun()

    st.divider()

    st.subheader("Pliki konfiguracji i backupy")
    # download users.json (backup z bazy)
    if st.button("Przygotuj backup users.json do pobrania"):
        try:
            st.download_button(
                "Kliknij aby pobrać users.json",
                data=json.dumps(db, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="users_backup.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(f"Błąd: {e}")

    # upload new tasks.json (replace)
    st.markdown("**Zastąp plik data/tasks.json (upload)**")
    uploaded_tasks = st.file_uploader(
        "Wgraj tasks.json (zastąpi obecny)", type=["json"], key="admin_upload_tasks"
    )
    if uploaded_tasks is not None:
        try:
            new_tasks = json.load(uploaded_tasks)
            tf = os.path.join(DATA_DIR, "tasks.json")
            with open(tf, "w", encoding="utf-8") as f:
                json.dump(new_tasks, f, ensure_ascii=False, indent=2)
            st.success("Zapisano data/tasks.json")
        except Exception as e:
            st.error(f"Błąd zapisu: {e}")

    # download tasks.json
    if st.button("Pobierz obecny data/tasks.json"):
        tf = os.path.join(DATA_DIR, "tasks.json")
        if os.path.exists(tf):
            with open(tf, "r", encoding="utf-8") as f:
                content = f.read()
            st.download_button(
                "Pobierz tasks.json",
                data=content.encode("utf-8"),
                file_name="tasks.json",
                mime="application/json",
            )
        else:
            st.info("Brak pliku data/tasks.json")

    st.divider()

    st.subheader("🎁 Konkursy i losowanie nagród")

    donors = _load_donors()
    draws = _load_draws()

    st.caption(f"Zgłoszeń konkursowych: {len(donors)}")

    if not donors:
        st.info(
            "Brak zgłoszeń — najpierw niech rodzice wypełnią formularz "
            "w zakładce 'Wsparcie & konkursy'."
        )
    else:
        show_donors = st.checkbox("Pokaż listę zgłoszeń", value=False)
        if show_donors:
            try:
                df_donors = pd.DataFrame(donors)
                st.dataframe(df_donors, use_container_width=True)
            except Exception:
                st.json(donors)

        st.markdown("#### Konfiguracja losowania")

        max_winners = max(1, len(donors))
        num_winners = st.number_input(
            "Liczba zwycięzców do wylosowania",
            min_value=1,
            max_value=max_winners,
            value=min(3, max_winners),
            step=1,
        )

        mode = st.radio(
            "Sposób liczenia losów:",
            [
                "Każde zgłoszenie = 1 los",
                "Unikalny e-mail = 1 los",
            ],
            index=0,
            help=(
                "Każde zgłoszenie = ktoś kto zrobił kilka wpłat ma kilka losów.\n"
                "Unikalny e-mail = każdy kontakt ma tylko jeden los."
            ),
        )

        if st.button("🎲 Wylosuj zwycięzców"):
            import random

            pool = donors
            if mode == "Unikalny e-mail = 1 los":
                uniq = {}
                for d in donors:
                    key = d.get("contact") or ""
                    if key and key not in uniq:
                        uniq[key] = d
                pool = list(uniq.values())

            if not pool:
                st.warning("Brak prawidłowych zgłoszeń z e-mailem do losowania.")
            else:
                k = min(num_winners, len(pool))
                winners = random.sample(pool, k=k)

                st.success(f"Wylosowano {k} zwycięzców:")
                st.json(winners)

                draw_record = {
                    "timestamp": datetime.now(tz=tz.gettz("Europe/Warsaw")).isoformat(),
                    "mode": mode,
                    "num_candidates": len(pool),
                    "num_winners": k,
                    "winners": winners,
                }
                draws.append(draw_record)
                _save_draws(draws)
                st.info("Zapisano wynik losowania do historii.")

    if draws:
        st.markdown("#### Historia losowań")
        with st.expander("Pokaż historię losowań"):
            try:
                df_draws = pd.DataFrame(
                    [
                        {
                            "czas": d.get("timestamp"),
                            "tryb": d.get("mode"),
                            "kandydaci": d.get("num_candidates"),
                            "zwycięzcy": ", ".join(
                                f"{w.get('parent_name','?')} <{w.get('contact','?')}>"
                                for w in d.get("winners", [])
                            ),
                        }
                        for d in draws
                    ]
                )
                st.dataframe(df_draws, use_container_width=True)
            except Exception:
                st.json(draws)

            st.download_button(
                "Pobierz historię losowań (JSON)",
                data=json.dumps(draws, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="draws.json",
                mime="application/json",
            )

    st.divider()
    st.subheader("Ustawienia PIN rodzica")
    admin_action = st.radio(
        "Akcja", ["Pokaż rekord PIN rodzica", "Resetuj PIN rodzica"], index=0
    )
    if admin_action == "Pokaż rekord PIN rodzica":
        rec = db.get("_parent_pin", {})
        st.json(rec)
        st.caption("To tylko rekord (salt + hash). Nie da się odtworzyć pierwotnego PINu z hash.")
    else:
        if st.button("Resetuj PIN rodzica do domyślnego 1234"):
            salt = secrets.token_hex(16)
            db["_parent_pin"] = {"salt": salt, "hash": hash_text(salt + "1234")}
            _save_users(db)
            st.success("Zresetowano PIN rodzica do 1234 (zmień go przez Panel rodzica).")

    st.divider()
    st.subheader("Ustawienia admina")
    if st.button("Obróć sekret TOTP (wymaga ponownego ustawienia w Authenticator)"):
        new_secret = pyotp.random_base32()
        set_admin_totp_secret(new_secret)
        st.success(
            "Wygenerowano nowy sekret. Zeskanuj nowy QR lub użyj secretu w sekcji powyżej. "
            "Bieżąca sesja została wylogowana."
        )
        st.session_state["admin_unlocked"] = False
        st.experimental_rerun()

    st.markdown("Koniec panelu administratora.")



# -----------------------------
# PANEL RODZICA
# -----------------------------

elif page == "Panel rodzica":
    st.markdown(f"<div class='big-title'>{PARENT_EMOJI} Panel rodzica</div>", unsafe_allow_html=True)

    # Auto-unlock on Enter
    if not st.session_state.get("parent_unlocked", False):
        st.markdown("Wpisz PIN, by odblokować ustawienia:")
        st.text_input(
            "PIN rodzica",
            type="password",
            key="parent_pin_input",
            on_change=_try_unlock_parent,
        )
        st.info("Wpisz PIN i naciśnij Enter.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Raport", "Dane i prywatność", "Ustawienia PIN"])

    # === TAB 1: RAPORT AKTYWNOŚCI ===
    with tab1:
        st.subheader("Raport aktywności")

        # --- Główne metryki ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Poziom", current_level(st.session_state.xp))
        c2.metric("XP", st.session_state.xp)
        c3.metric("Odznaki", len(st.session_state.badges))
        c4.metric("Naklejki", len(st.session_state.stickers))

        events = st.session_state.activity_log

        if events:
            # --- DataFrame z logów ---
            df = pd.DataFrame(events)  # kolumny: time, event

            # parsowanie czasu
            try:
                df["time_dt"] = pd.to_datetime(df["time"])
            except Exception:
                df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")

            df = df.dropna(subset=["time_dt"])

            if not df.empty:
                df["day"] = df["time_dt"].dt.date
                # kategoria = pierwsza część eventu przed "_"
                df["category"] = df["event"].str.split("_").str[0]

                st.markdown("### 📈 Aktywność w czasie")

                # liczba zdarzeń dziennie
                daily_counts = (
                    df.groupby("day")
                    .size()
                    .reset_index(name="liczba_zdarzeń")
                    .sort_values("day")
                )

                if not daily_counts.empty:
                    chart_daily = (
                        alt.Chart(daily_counts)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("day:T", title="Dzień"),
                            y=alt.Y("liczba_zdarzeń:Q", title="Liczba zdarzeń"),
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(chart_daily, use_container_width=True)
                else:
                    st.caption("Brak danych do wykresu dziennej aktywności.")

                st.markdown("### 📊 Typy aktywności")

                cat_counts = (
                    df["category"]
                    .value_counts()
                    .reset_index()
                    .rename(columns={"index": "category", "category": "count"})
                )

                if not cat_counts.empty:
                    chart_cat = (
                        alt.Chart(cat_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X("category:N", title="Kategoria"),
                            y=alt.Y("count:Q", title="Liczba zdarzeń"),
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(chart_cat, use_container_width=True)
                else:
                    st.caption("Brak danych do wykresu kategorii zdarzeń.")

                # --- 🧠 Podsumowanie liczbowe ---
                st.markdown("### 🧠 Podsumowanie")

                total_events = len(df)
                most_active_day = (
                    daily_counts.sort_values("liczba_zdarzeń", ascending=False)
                    .iloc[0]["day"]
                    if not daily_counts.empty
                    else None
                )
                top_cat = (
                    cat_counts.iloc[0]["category"]
                    if not cat_counts.empty
                    else None
                )

                bullets = []
                bullets.append(
                    f"• Łączna liczba zarejestrowanych zdarzeń: **{total_events}**."
                )
                if most_active_day:
                    bullets.append(
                        f"• Najbardziej aktywny dzień: **{most_active_day}** "
                        f"({int(daily_counts.iloc[0]['liczba_zdarzeń'])} zdarzeń)."
                    )
                if top_cat:
                    bullets.append(
                        f"• Najczęstszy typ aktywności (kategoria eventu): **{top_cat}**."
                    )

                for b in bullets:
                    st.write(b)

                # --- 🤖 Podpowiedzi Data4Kids ---
                st.markdown("### 🤖 Podpowiedzi Data4Kids")

                from datetime import datetime, timezone

                tips = []

                # 1) Jak dawno dziecko było aktywne?
                now = datetime.now(timezone.utc)
                last_time = df["time_dt"].max()
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)

                days_since = (now - last_time).days
                if days_since >= 7:
                    tips.append(
                        f"• Ostatnia aktywność była **ponad {days_since} dni temu**. "
                        "Może warto zaplanować wspólną sesję z Data4Kids w najbliższy weekend? 🙂"
                    )
                elif days_since >= 3:
                    tips.append(
                        f"• Ostatnia aktywność była **{days_since} dni temu**. "
                        "Drobna przerwa jest OK, ale delikatne przypomnienie może pomóc."
                    )
                else:
                    tips.append(
                        "• Dziecko korzysta z aplikacji **regularnie w ostatnich dniach** – super! 🚀"
                    )

                # 2) Quizy – poprawne vs błędne
                df["is_quiz_ok"] = df["event"].str.contains("quiz_ok")
                df["is_quiz_fail"] = df["event"].str.contains("quiz_fail")

                quiz_ok = int(df["is_quiz_ok"].sum())
                quiz_fail = int(df["is_quiz_fail"].sum())

                if quiz_ok + quiz_fail > 0:
                    fail_rate = quiz_fail / (quiz_ok + quiz_fail)
                    if fail_rate > 0.5 and quiz_fail >= 3:
                        tips.append(
                            "• W ostatnich quizach jest **sporo błędnych odpowiedzi**. "
                            "Może warto potraktować je jako okazję do rozmowy, a nie oceniania? 🙂"
                        )
                    elif quiz_ok >= 3 and fail_rate < 0.3:
                        tips.append(
                            "• Dziecko **radzi sobie bardzo dobrze w quizach** – "
                            "można pomyśleć o trudniejszych misjach lub nowych wyzwaniach."
                        )
                else:
                    tips.append(
                        "• Brak danych z quizów – spróbuj zachęcić do wykonania choć jednego quizu, "
                        "żeby zobaczyć mocne strony dziecka."
                    )

                # 3) Dominujące typy aktywności (na bazie category)
                if not cat_counts.empty:
                    top_cat_name = cat_counts.iloc[0]["category"]
                    if top_cat_name == "school":
                        tips.append(
                            "• Najczęściej wykonywane są zadania z **Przedmiotów szkolnych**. "
                            "To świetne uzupełnienie nauki w szkole. 📚"
                        )
                    elif top_cat_name == "quiz":
                        tips.append(
                            "• Dziecko najczęściej wybiera **quizy** – lubi szybkie sprawdzanie wiedzy. "
                            "Można dorzucić misje fabularne dla urozmaicenia. 🎭"
                        )
                    elif top_cat_name == "image":
                        tips.append(
                            "• Dużo aktywności w **quizach obrazkowych** – "
                            "to dobra okazja do rozmów o emocjach i spostrzegawczości. 😊"
                        )
                    elif top_cat_name == "dataset":
                        tips.append(
                            "• Często używane są **dane i wykresy** – "
                            "to świetne budowanie myślenia analitycznego. 📊"
                        )

                if tips:
                    for t in tips:
                        st.write(t)
                else:
                    st.caption(
                        "Brak szczegółowych podpowiedzi – potrzeba więcej danych."
                    )

                # --- 🧪 Diagnoza quizów ---
                st.markdown("### 🧪 Diagnoza quizów")

                quiz_df = df[df["is_quiz_ok"] | df["is_quiz_fail"]].copy()

                if quiz_df.empty:
                    st.caption(
                        "Brak danych z quizów – diagnoza pojawi się po kilku próbach quizów."
                    )
                else:
                    # 1) Procent poprawnych odpowiedzi w czasie
                    quiz_daily = (
                        quiz_df.groupby("day")
                        .agg(
                            ok=("is_quiz_ok", "sum"),
                            total=("is_quiz_ok", "size"),
                        )
                        .reset_index()
                    )
                    quiz_daily["percent_ok"] = (
                        quiz_daily["ok"] / quiz_daily["total"] * 100
                    ).round(1)

                    if not quiz_daily.empty:
                        st.markdown("#### 📈 Procent poprawnych odpowiedzi w czasie")
                        chart_quiz = (
                            alt.Chart(quiz_daily)
                            .mark_line(point=True)
                            .encode(
                                x=alt.X("day:T", title="Dzień"),
                                y=alt.Y(
                                    "percent_ok:Q",
                                    title="% poprawnych odpowiedzi",
                                    scale=alt.Scale(domain=[0, 100]),
                                ),
                                tooltip=["day", "ok", "total", "percent_ok"],
                            )
                            .properties(height=250)
                        )
                        st.altair_chart(chart_quiz, use_container_width=True)
                        st.caption(
                            "Wykres pokazuje, jak zmienia się skuteczność odpowiedzi w quizach w czasie."
                        )

                    # 2) Top najtrudniejsze typy pytań
                    def _parse_quiz_event(ev: str):
                        parts = str(ev).split("::")
                        if not parts or parts[0] not in ("quiz_ok", "quiz_fail"):
                            return None
                        status = "ok" if parts[0] == "quiz_ok" else "fail"
                        source = parts[1] if len(parts) > 1 else "inne"

                        if source == "data":
                            qid = parts[2] if len(parts) > 2 else None
                            short_q = parts[3] if len(parts) > 3 else ""
                            category = "Quiz danych"
                            wrong = parts[4] if status == "fail" and len(parts) > 4 else None
                            correct = parts[5] if status == "fail" and len(parts) > 5 else None
                        elif source == "image":
                            image_cat = parts[2] if len(parts) > 2 else "inne"
                            qid = parts[3] if len(parts) > 3 else None
                            short_q = parts[4] if len(parts) > 4 else ""
                            mapping = {
                                "emotions": "Emocje",
                                "shapes": "Kształty",
                                "plots": "Wykresy",
                                "objects": "Przedmioty",
                            }
                            category = mapping.get(image_cat, image_cat)
                            wrong = parts[5] if status == "fail" and len(parts) > 5 else None
                            correct = parts[6] if status == "fail" and len(parts) > 6 else None
                        else:
                            qid = parts[2] if len(parts) > 2 else None
                            short_q = parts[3] if len(parts) > 3 else ""
                            category = source
                            wrong = parts[4] if status == "fail" and len(parts) > 4 else None
                            correct = parts[5] if status == "fail" and len(parts) > 5 else None

                        return {
                            "status": status,
                            "source": source,
                            "category": category,
                            "qid": qid,
                            "short_q": short_q,
                            "wrong": wrong,
                            "correct": correct,
                        }

                    parsed_rows = []
                    for ev in quiz_df["event"]:
                        parsed = _parse_quiz_event(ev)
                        if parsed:
                            parsed_rows.append(parsed)

                    if parsed_rows:
                        df_parsed = pd.DataFrame(parsed_rows)

                        # statystyki per kategoria
                        cat_stats = (
                            df_parsed.groupby("category")
                            .agg(
                                total=("status", "size"),
                                wrong=("status", lambda s: (s == "fail").sum()),
                                ok=("status", lambda s: (s == "ok").sum()),
                            )
                            .reset_index()
                        )
                        cat_stats["fail_pct"] = (
                            cat_stats["wrong"] / cat_stats["total"] * 100
                        ).round(1)

                        hard_cats = (
                            cat_stats[cat_stats["total"] >= 3]
                            .sort_values(
                                ["fail_pct", "total"], ascending=[False, False]
                            )
                        )

                        if not hard_cats.empty:
                            st.markdown("#### 🧩 Top najtrudniejsze typy pytań")
                            for _, row in hard_cats.head(3).iterrows():
                                st.markdown(
                                    f"- **{row['category']}** – błędne odpowiedzi: "
                                    f"{int(row['wrong'])} / {int(row['total'])} "
                                    f"({row['fail_pct']}%)."
                                )
                        else:
                            st.caption(
                                "Za mało odpowiedzi, żeby policzyć najtrudniejsze typy pytań."
                            )

                        # Najczęstsza pomyłka (np. emocje – smutek vs strach)
                        hard_pairs = df_parsed[df_parsed["status"] == "fail"].dropna(
                            subset=["wrong", "correct"]
                        )
                        if not hard_pairs.empty:
                            pair_stats = (
                                hard_pairs.groupby(
                                    ["category", "wrong", "correct"]
                                )
                                .size()
                                .reset_index(name="count")
                                .sort_values("count", ascending=False)
                            )
                            top_pair = pair_stats.iloc[0]
                            st.caption(
                                f"Najczęstsza pomyłka: **{top_pair['category']} – "
                                f"„{top_pair['wrong']}” zamiast „{top_pair['correct']}”** "
                                f"({int(top_pair['count'])}×)."
                            )
                    else:
                        st.caption(
                            "Na razie brak szczegółowych danych o tym, które pytania sprawiają trudność."
                        )

                # --- ostatnie surowe logi ---
                st.markdown("### 📜 Ostatnie działania")
                last_events = events[-10:][::-1]
                for e in last_events:
                    st.write(f"• {e['time']} — {e['event']}")

            else:
                st.caption(
                    "Brak zarejestrowanych zdarzeń — raport pojawi się po pierwszych aktywnościach."
                )
        else:
            st.caption("Brak zdarzeń — zacznij od strony Start lub Misje.")

        # --- Szczegółowy raport JSON (jak było) ---
        with st.expander("Pokaż szczegóły (JSON)"):
            overview = {
                "app": APP_NAME,
                "version": VERSION,
                "kid_name": st.session_state.kid_name or "(bez imienia)",
                "age": st.session_state.age,
                "age_group": st.session_state.age_group,
                "timestamp": datetime.now(tz=tz.gettz("Europe/Warsaw")).isoformat(),
                "events": st.session_state.activity_log[-100:],
                "data_shape": list(st.session_state.data.shape),
                "xp": st.session_state.xp,
                "level": current_level(st.session_state.xp),
                "badges": sorted(list(st.session_state.badges)),
                "stickers": sorted(list(st.session_state.stickers)),
            }
            st.json(overview)
            st.download_button(
                "Pobierz raport JSON",
                data=json.dumps(overview, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="data4kids_raport.json",
                mime="application/json",
            )

    # === TAB 2 i TAB 3 zostają tak jak były (Dane i prywatność, Ustawienia PIN) ===



# -----------------------------
# Footer

# -----------------------------
st.markdown(
    f"<span class='muted'>v{VERSION} — {APP_NAME}. Zrobione z ❤️ w Streamlit. "
    f"<span class='pill kid'>daily quest</span> <span class='pill kid'>misje</span> "
    f"<span class='pill kid'>symulacje</span> <span class='pill kid'>czyszczenie</span> "
    f"<span class='pill kid'>fabuła</span> <span class='pill kid'>przedmioty</span> "
    f"<span class='pill kid'>album</span> <span class='pill kid'>quizy</span> "
    f"<span class='pill parent'>panel rodzica</span></span>",
    unsafe_allow_html=True,
)
