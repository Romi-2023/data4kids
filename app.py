
import os
import json
import hashlib
import secrets
import random
from math import ceil
from datetime import datetime, date
from dateutil import tz
from typing import Optional, List, Dict

import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
import numpy as np

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
DATA_DIR = os.environ.get("D4K_DATA_DIR", "/tmp/data4kids")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")
DONORS_FILE = os.path.join(DATA_DIR, "donors.json")  # NOWE
DONORS_FILE = os.path.join(DATA_DIR, "donors.json")  # zgłoszenia do konkursów
DRAWS_FILE = os.path.join(DATA_DIR, "draws.json")    # historia losowań

def _load_donors():
    if not os.path.exists(DONORS_FILE):
        return []
    try:
        with open(DONORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_donors(records: list) -> None:
    with open(DONORS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def _load_draws():
    if not os.path.exists(DRAWS_FILE):
        return []
    try:
        with open(DRAWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_draws(records: list) -> None:
    with open(DRAWS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_users(db: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


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

# Glossary
GLOSSARY = {
    "średnia": "Suma wszystkich wartości podzielona przez ich liczbę.",
    "mediana": "Wartość środkowa po ułożeniu danych od najmniejszej do największej.",
    "korelacja": "Miara tego, jak dwie rzeczy zmieniają się razem (dodatnia, ujemna, brak).",
    "agregacja": "Łączenie danych (np. liczenie średniej) w grupach.",
    "kategoria": "Słowo/etykieta zamiast liczby (np. kolor, miasto).",
}
COUNT_LABEL = "liczba osób"

# --- Categorized kid-friendly glossary (used only on Słowniczek page) ---
CATEGORIZED_GLOSSARY = {
  "MATEMATYKA": {
    "parzysta liczba": "Dzieli się przez 2 bez reszty (np. 4, 10, 28).",
    "nieparzysta liczba": "Nie dzieli się przez 2 bez reszty (np. 3, 7, 19).",
    "dzielnik": "Liczba, przez którą dzielimy inną liczbę.",
    "wielokrotność": "Wynik mnożenia danej liczby (np. wielokrotności 5 to 10, 15, 20…).",
    "liczba pierwsza": "Ma dokładnie dwa dzielniki: 1 i samą siebie (np. 2, 3, 5, 7).",
    "ułamek": "Część całości zapisana jak 1/2, 3/4.",
    "ułamek dziesiętny": "Ułamek zapisany z przecinkiem (np. 0,5).",
    "procent": "Część ze 100, np. 25% to 25 na 100.",
    "pole figury": "Powierzchnia w środku figury (np. ile farby by zakryło kształt).",
    "obwód": "Długość dookoła figury.",
    "kąt prosty": "Ma 90°.",
    "średnia arytmetyczna": "Suma liczb podzielona przez ich liczbę.",
    "promień": "Od środka okręgu do jego brzegu.",
    "średnica": "Od brzegu do brzegu przez środek (2× promień).",
    "proporcja": "Porównanie dwóch wielkości tak, by zachować ten sam stosunek.",
    "prędkość": "Jak szybko coś się porusza (v = s/t)."
  },
  "POLSKI": {
    "rzeczownik": "Nazywa osoby, zwierzęta, rzeczy (np. kot, szkoła).",
    "czasownik": "Mówi co się dzieje (np. biega, czyta).",
    "przymiotnik": "Opisuje cechę (np. szybki, zielona).",
    "przysłówek": "Opisuje czynność (np. szybko, cicho).",
    "podmiot": "Kto/co wykonuje czynność w zdaniu.",
    "orzeczenie": "Co robi podmiot (czasownik w zdaniu).",
    "epitet": "Słowo ozdabiające rzeczownik (np. srebrny księżyc).",
    "antonim": "Słowo przeciwne (wysoki ↔ niski).",
    "synonim": "Słowo podobne znaczeniem (ważny ↔ istotny).",
    "rym": "Podobne brzmienia na końcach wyrazów (kotek – płotek).",
    "narrator": "Głos opowiadający historię w tekście."
  },
  "HISTORIA": {
    "średniowiecze": "Czas między starożytnością a nowożytnością.",
    "konstytucja": "Najważniejsze prawo państwa.",
    "unia lubelska": "Połączenie Polski i Litwy w 1569 roku.",
    "zabory": "Podział Polski przez sąsiadów w XVIII wieku.",
    "powstanie": "Wystąpienie zbrojne przeciw władzy.",
    "rycerz": "Wojownik konny z dawnych czasów.",
    "dynastia": "Ród panujący przez wiele pokoleń."
  },
  "GEOGRAFIA": {
    "kontynent": "Ogromny ląd, np. Afryka, Europa.",
    "ocean": "Bardzo wielka masa słonej wody.",
    "pustynia": "Miejsce z małą ilością opadów (np. Sahara).",
    "wyżyna": "Dość wysokie, rozległe tereny.",
    "nizina": "Płaski, niski teren.",
    "delta": "Rozgałęzienie rzeki przy ujściu do morza.",
    "klimat": "Typ pogody w danym miejscu przez długi czas.",
    "wulkan": "Góra, z której może wydobywać się lawa."
  },
  "FIZYKA": {
    "siła": "Oddziaływanie, które może zmieniać ruch lub kształt.",
    "masa": "Ilość materii w ciele.",
    "ciśnienie": "Siła nacisku na powierzchnię (p=F/S).",
    "energia": "Zdolność do wykonania pracy.",
    "praca": "Przekaz energii przez działanie siłą na odcinku.",
    "gęstość": "Masa w danej objętości (ρ=m/V).",
    "tarcie": "Siła hamująca ruch przy dotyku powierzchni.",
    "prędkość światła": "Około 300 000 km/s."
  },
  "CHEMIA": {
    "atom": "Najmniejsza cząstka pierwiastka.",
    "pierwiastek": "Substancja złożona z jednakowych atomów (np. tlen).",
    "związek chemiczny": "Połączenie co najmniej dwóch pierwiastków (np. H₂O).",
    "mieszanina": "Połączenie substancji bez reakcji chemicznej.",
    "roztwór": "Jednorodna mieszanina, np. sól w wodzie.",
    "kwas": "Ma pH < 7 (np. sok z cytryny jest kwaśny).",
    "zasada": "Ma pH > 7 (np. mydło jest zasadowe).",
    "pH": "Skala kwasowości od 0 do 14."
  },
  "ANGIELSKI": {
    "noun": "Rzeczownik.",
    "verb": "Czasownik.",
    "adjective": "Przymiotnik.",
    "adverb": "Przysłówek.",
    "plural": "Liczba mnoga.",
    "past simple": "Czas przeszły prosty (went, saw).",
    "present simple": "Czas teraźniejszy prosty (go, see).",
    "present continuous": "Czynność trwająca teraz (is/are + -ing).",
    "future simple": "Czas przyszły prosty (will + bezokolicznik).",
    "to be": "Czasownik 'być' (am/is/are).",
    "to have": "Mieć (have/has).",
    "irregular verbs": "Czasowniki nieregularne (go–went–gone…).",
    "question": "Pytanie.",
    "sentence": "Zdanie."
  },
    "NIEMIECKI": {
      "der Hund": "pies.",
      "die Katze": "kot.",
      "die Schule": "szkoła.",
      "das Haus": "dom.",
      "die Stadt": "miasto.",
      "die Zahl": "liczba.",
      "lesen": "czytać.",
      "schreiben": "pisać.",
      "sprechen": "mówić.",
      "hören": "słuchać.",
      "gut": "dobry, dobrze.",
      "schön": "ładny, piękny.",
      "schnell": "szybki, szybko.",
      "langsam": "wolny, powoli.",
      "Hallo": "cześć.",
      "Tschüss": "pa, na razie.",
      "Bitte": "proszę (np. podając coś / jako „proszę bardzo”).",
      "Danke": "dziękuję."
  },
  "BIOLOGIA": {
    "komórka": "Najmniejsza część żywego organizmu.",
    "tkanka": "Zespół podobnych komórek.",
    "narząd": "Część ciała z określoną funkcją (np. serce).",
    "układ oddechowy": "Służy do oddychania (płuca, tchawica).",
    "układ krążenia": "Transportuje krew (serce, naczynia).",
    "fotosynteza": "Rośliny tworzą pokarm z wody, dwutlenku węgla i światła.",
    "DNA": "Instrukcja życia zapisana w komórkach.",
    "chlorofil": "Zielony barwnik w roślinach."
  },
  "DANE I STATYSTYKA": {
    "dane": "Zebrane informacje, liczby, odpowiedzi.",
    "średnia": "Suma podzielona przez liczbę elementów.",
    "mediana": "Środkowa wartość po ułożeniu od najmniejszej do największej.",
    "moda": "Wartość, która występuje najczęściej.",
    "wykres słupkowy": "Wysokość słupków pokazuje wartości.",
    "wykres kołowy": "Koło podzielone na kawałki pokazujące części całości.",
    "ankieta": "Pytania, które zbierają odpowiedzi od ludzi."
  }
}

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
    return val in ("count()", COUNT_LABEL)

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

def get_leaderboard(limit: int = 10) -> List[Dict]:
    """Prosty ranking po XP – baza pod konkursy."""
    db = _load_users()
    rows = []
    for name, profile in db.items():
        if name.startswith("_"):
            continue  # pomijamy rekordy techniczne
        rows.append({
            "user": name,
            "xp": int(profile.get("xp", 0)),
            "badges": len(profile.get("badges", [])),
            "stickers": len(profile.get("stickers", [])),
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

def safe_load_json(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

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
            "Hall of Fame",
            "Wsparcie & konkursy",
            "Regulamin",
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
PUBLIC_PAGES = {"Start", "Regulamin", "Administrator", "Panel rodzica", "Wsparcie & konkursy"}

if page not in PUBLIC_PAGES and not st.session_state.get("user"):
    st.info("Najpierw zaloguj się na stronie **Start**. Potem możesz korzystać z całej aplikacji. 🚀")
    st.stop()

# -----------------------------
# START (with auth gate)
# -----------------------------
if page == "Start":
    ...


# -----------------------------
# START (with auth gate)
# -----------------------------
if page == "Start":
    st.markdown("### 🔐 Logowanie")

    auth_tab_login, auth_tab_reg = st.tabs(["Zaloguj", "Zarejestruj"])
    db = _load_users()

    with auth_tab_login:
        li_user = st.text_input("Login", key="li_user")
        li_pass = st.text_input("Hasło", type="password", key="li_pass")
        if st.button("Zaloguj"):
            if li_user in db:
                salt = db[li_user]["salt"]
                if hash_pw(li_pass, salt) == db[li_user]["password_hash"]:
                    st.session_state.user = li_user
                    st.session_state.xp = int(db[li_user].get("xp", 0))
                    st.session_state.stickers = set(db[li_user].get("stickers", []))
                    st.session_state.badges = set(db[li_user].get("badges", []))
                    st.success(f"Zalogowano jako **{li_user}** 🎉")
                else:
                    st.error("Błędne hasło.")
            else:
                st.error("Taki login nie istnieje.")

    with auth_tab_reg:
        re_user = st.text_input("Nowy login", key="re_user")
        re_pass = st.text_input("Hasło", type="password", key="re_pass")
        re_pass2 = st.text_input("Powtórz hasło", type="password", key="re_pass2")
        if st.button("Zarejestruj"):
            if not re_user or not re_pass:
                st.error("Podaj login i hasło.")
            elif re_user in db:
                st.error("Taki login już istnieje.")
            elif re_pass != re_pass2:
                st.error("Hasła się różnią.")
            else:
                salt = secrets.token_hex(8)
                db[re_user] = {"salt": salt, "password_hash": hash_pw(re_pass, salt), "xp": 0, "stickers": [], "badges": []}
                _save_users(db)
                st.success("Utworzono konto! Teraz zaloguj się zakładką 'Zaloguj'.")

    if not st.session_state.user:
        st.info("Zaloguj się, aby kontynuować.")
        st.stop()

    # rest of Start
    st.markdown(f"<div class='big-title'>🧒 {KID_EMOJI} Witaj w {APP_NAME}!</div>", unsafe_allow_html=True)
    colA, colB = st.columns([1, 1])
    with colA:
        st.text_input("Twoje imię (opcjonalnie)", key="kid_name")
        age_in = st.number_input("Ile masz lat?", min_value=7, max_value=14, step=1, value=10)
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
            st.session_state.data = make_dataset(n, cols, seed=random.randint(1, 999999))
            st.success(f"Załadowano: {preset_name}")
            log_event(f"dataset_loaded_{group}_{preset_name}")

        if st.button("Start misji 🚀"):
            log_event(f"kid_started_{group}")
            st.success("Super! Wejdź do »Misje« i działamy.")

    with colB:
        st.write("""
        **Co zrobimy?**
        - Daily Quest ✅
        - Rysowanie, detektyw 🕵️
        - Symulacje 🎲, Czyszczenie ✍️, Fabuła 📖
        - Przedmioty szkolne 📚 (mat, pol, hist, geo, fiz, chem, ang)
        - Album naklejek 🗂️ i Quizy 🖼️🧠
        - XP, odznaki i poziomy 🔓, Hall of Fame 🏆
        """)
        st.markdown(
            f"XP: **{st.session_state.xp}** | Poziom: **L{current_level(st.session_state.xp)}** "
            + "".join([f"<span class='badge'>🏅 {b}</span>" for b in st.session_state.badges]),
            unsafe_allow_html=True,
        )

# -----------------------------
# Pozostałe podstrony (skrócone do kluczowych)
# -----------------------------
elif page == "Poznaj dane":
    st.markdown(f"<div class='big-title'>📊 {KID_EMOJI} Poznaj dane</div>", unsafe_allow_html=True)
    df_base = st.session_state.data.copy()
    N = min(15, len(df_base)) if len(df_base) else 0
    df_daily = pick_daily_sample(df_base, n=max(1, N)) if N else df_base
    fantasy_mode = st.session_state.get("fantasy_mode", True)
    df_view = apply_fantasy(df_daily) if fantasy_mode else df_daily

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Liczba wierszy (zestaw dnia)", len(df_view))
    if "wiek" in df_view.columns: c2.metric("Śr. wiek", round(pd.to_numeric(df_view["wiek"], errors='coerce').mean(), 1))
    if "wzrost_cm" in df_view.columns: c3.metric("Śr. wzrost (cm)", round(pd.to_numeric(df_view["wzrost_cm"], errors='coerce').mean(), 1))
    if "miasto" in df_view.columns: c4.metric("Miasta", df_view["miasto"].nunique())
    with st.expander("Zobacz tabelę"):
        st.caption(f"Zestaw dzienny: {date.today().isoformat()}")
        st.dataframe(df_view.head(50), width='stretch')

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
    st.markdown(f"<div class='big-title'>🗺️ {KID_EMOJI} Misje</div>", unsafe_allow_html=True)
    missions_path = os.path.join(DATA_DIR, "missions.json")
    missions = safe_load_json(missions_path, default=[])
    if not missions:
        st.info("Brak misji. Dodaj je do data/missions.json")
    else:
        for m in missions:
            with st.expander(f"🎯 {m.get('title','Misja')} (+{m.get('reward_xp',10)} XP)"):
                st.write(m.get("desc",""))
                st.caption("Kroki: " + ", ".join(m.get("steps", [])))
                if st.button("Oznacz jako ukończoną ✅", key=f"mis_{m.get('id','x')}"):
                    st.success("Zaliczone! +XP przyznane (symbolicznie).")

elif page == "Quiz danych":
    st.markdown(f"<div class='big-title'>📊 {KID_EMOJI} Quiz danych</div>", unsafe_allow_html=True)
    dq_path = os.path.join(DATA_DIR, "quizzes", "data_quiz.json")
    dq = safe_load_json(dq_path, default={"items":[]})
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
        q = t["q"]; opts = t["options"]; corr = int(t["correct"])
        st.markdown(f"**{i}. {q}**")
        choice = st.radio("Wybierz:", opts, key=f"dq_{i}", label_visibility="collapsed", index=None)
        if st.button("Sprawdź ✅", key=f"dq_check_{i}"):
            if choice is None:
                st.warning("Wybierz odpowiedź.")
            elif opts.index(choice) == corr:
                st.success("✅ Dobrze!")
            else:
                st.error(f"❌ Nie. Poprawna: **{opts[corr]}**.")

elif page == "Quiz obrazkowy":
    st.markdown(f"<div class='big-title'>🖼️ {KID_EMOJI} Quiz obrazkowy</div>", unsafe_allow_html=True)

    iq_path = os.path.join(DATA_DIR, "quiz_images", "image_quiz.json")
    iq = safe_load_json(iq_path, default={"items": []})
    items = iq.get("items", [])

    # policz wszystkie pytania ze wszystkich obrazków
    total_q = sum(len(item.get("questions", [])) for item in items)
    st.caption(f"Liczba pytań: {total_q}")

    q_counter = 0  # globalny licznik pytań

    for img_idx, item in enumerate(items, start=1):
        img_path = item.get("image")
        questions = item.get("questions", [])

        if not questions:
            continue

        # obrazek wyświetlamy raz dla całego zestawu pytań
        try:
            st.image(img_path, caption=f"Obrazek {img_idx}", use_container_width=True)
        except Exception:
            st.caption(f"(Brak obrazu: {img_path})")

        for local_q_idx, t in enumerate(questions, start=1):
            q_counter += 1
            q = t.get("q", "")
            opts = t.get("options", [])
            corr = int(t.get("correct", 0))

            st.markdown(f"**{q_counter}. {q}**")

            key_base = f"iq_{img_idx}_{local_q_idx}"
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
                elif opts and opts.index(choice) == corr:
                    st.success("✅ Dobrze!")
                else:
                    if opts:
                        st.error(f"❌ Nie. Poprawna: **{opts[corr]}**.")
                    else:
                        st.error("Brak opcji odpowiedzi w danych quizu.")


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
            sub_teoria, sub_znaki, sub_quiz = st.tabs(["Teoria", "Znaki", "Quiz"])

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

                            # ŚCIEŻKA DO OBRAZKA – TU JEST MAGIA :)
                            img_file = os.path.join("rower_signs", f"{code}.png")


                            if os.path.exists(img_file):
                                st.image(img_file, width=140)
                            else:
                                st.caption(f"(Brak obrazka: {img_file})")

                            st.markdown(f"**Opis:** {sign.get('description','')}")
                            st.markdown(f"**Przykład:** {sign.get('example','')}")

            with sub_quiz:
                items = quiz.get("questions", [])
                if not items:
                    st.info("Brak pytań w pliku quizu.")
                else:
                    # --- Wspólna dzienna pula pytań dla Nauki i Egzaminu ---
                    day_idx = days_since_epoch()
                    k_daily = min(10, len(items))  # ile pytań dziennie
                    daily_items = pick_daily_chunk(items, k_daily, day_idx, "rower_quiz")

                    if not daily_items:
                        st.info("Brak pytań w dzisiejszej puli.")
                        st.stop()

                    mode = st.radio(
                        "Tryb pracy:",
                        ["Nauka", "Egzamin próbny"],
                        horizontal=True,
                        key="rower_quiz_mode",
                    )

                    # === TRYB NAUKA ===
                    if mode == "Nauka":
                        st.caption(
                            f"Dzisiaj uczysz się na podstawie {len(daily_items)} pytań "
                            f"(z {len(items)} w całej bazie)."
                        )
                        for i, q in enumerate(daily_items, start=1):
                            st.markdown(f"**{i}. {q.get('question','')}**")
                            options = q.get("options", [])
                            if not options:
                                continue
                            correct_idx = int(q.get("correct", 0))
                            choice = st.radio(
                                "Wybierz odpowiedź:",
                                options,
                                key=f"rower_q_{i}",
                                label_visibility="collapsed",
                                index=None,
                            )
                            if st.button("Sprawdź", key=f"rower_q_check_{i}"):
                                if choice is None:
                                    st.warning("Najpierw wybierz odpowiedź.")
                                else:
                                    if options.index(choice) == correct_idx:
                                        st.success("✅ Dobrze!")
                                    else:
                                        st.error(
                                            f"❌ Nie, prawidłowa odpowiedź to: "
                                            f"**{options[correct_idx]}**."
                                        )
                                    expl = q.get("explanation")
                                    if expl:
                                        st.info(expl)

                    # === TRYB EGZAMIN PRÓBNY ===
                    else:
                        st.caption(
                            f"Egzamin próbny: dzisiejszy zestaw to {len(daily_items)} pytań "
                            f"(z {len(items)} w całej bazie)."
                        )

                        today_key = get_today_key()

                        # Jeśli weszliśmy w nowy dzień – resetujemy egzamin.
                        if st.session_state.get("rower_exam_date") != today_key:
                            st.session_state["rower_exam_initialized"] = False

                        if not st.session_state.get("rower_exam_initialized", False):
                            st.session_state["rower_exam_initialized"] = True
                            st.session_state["rower_exam_items"] = daily_items
                            st.session_state["rower_exam_current"] = 0
                            st.session_state["rower_exam_correct"] = 0
                            st.session_state["rower_exam_date"] = today_key

                        exam_items = st.session_state["rower_exam_items"]
                        cur = st.session_state["rower_exam_current"]

                        # Koniec egzaminu
                        if cur >= len(exam_items):
                            total = len(exam_items)
                            correct = st.session_state["rower_exam_correct"]
                            st.success(
                                f"Twój wynik: {correct} / {total} poprawnych odpowiedzi."
                            )
                            if st.button("Rozpocznij nowy egzamin"):
                                # Nowy egzamin tego samego dnia -> ta sama dzienna pula pytań
                                st.session_state["rower_exam_initialized"] = False
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
    subjects = ["matematyka","polski","historia","geografia","fizyka","chemia","angielski","niemiecki","biologia"]

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

    tab_math, tab_pol, tab_hist, tab_geo, tab_phys, tab_chem, tab_eng, tab_ger, tab_bio = st.tabs(
    ["Matematyka", "Język polski", "Historia", "Geografia", "Fizyka", "Chemia", "Angielski", "Niemiecki", "Biologia"]
)
    with tab_math: show_subject("matematyka", "Matematyka")
    with tab_pol:  show_subject("polski", "Język polski")
    with tab_hist: show_subject("historia", "Historia")
    with tab_geo:  show_subject("geografia", "Geografia")
    with tab_phys: show_subject("fizyka", "Fizyka")
    with tab_chem: show_subject("chemia", "Chemia")
    with tab_eng:  show_subject("angielski", "Angielski")
    with tab_ger:  show_subject("niemiecki", "Niemiecki")
    with tab_bio:  show_subject("biologia", "Biologia")

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
        # Przeglądanie kategorii
        tabs = st.tabs(list(CATEGORIZED_GLOSSARY.keys()))
        for (cat, entries), tab in zip(CATEGORIZED_GLOSSARY.items(), tabs):
            with tab:
                for i, (k, v) in enumerate(sorted(entries.items()), start=1):
                    cols = st.columns([3,1])
                    with cols[0]:
                        st.write(f"**{k}** — {v}")
                    with cols[1]:
                        if cat == "ANGIELSKI":
                            tts_button_en(k, key=f"{cat}_{i}")

elif page == "Hall of Fame":
    st.markdown("# 🏆 Hall of Fame")
    st.write("Dodaj swój profil do tabeli mistrzów i pobierz zaktualizowany plik JSON.")
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
        "missions_done": sorted([k for k, v in st.session_state.missions_state.items() if v.get("done")]),
    }
    st.subheader("Mój profil")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imię", st.session_state.kid_name or "—")
    c2.metric("Wiek", st.session_state.age or "—")
    c3.metric("Poziom", current_level(st.session_state.xp))
    c4.metric("XP", st.session_state.xp)
    st.caption(f"Odznaki: **{len(st.session_state.badges)}**  |  Naklejki: **{len(st.session_state.stickers)}**")
    st.download_button("Pobierz mój profil (JSON)", data=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="data4kids_profile.json", mime="application/json")

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
1. **Lokalnie, nie w chmurze.** Aplikacja działa na Twoim urządzeniu.  
   Nie wysyłamy danych na serwery i nie zbieramy analityki.

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



# -----------------------------
# ADMINISTRATOR (TOTP / Authenticator)
# -----------------------------
elif page == "Administrator":
    st.markdown("# 🛡️ Administrator")
    st.caption("Dostęp tylko przez TOTP (Authenticator) — sekret przechowywany lokalnie w data/users.json")

    # load/save admin TOTP secret in users DB under key "_admin_totp"
    db = _load_users()
    admin_rec = db.get("_admin_totp", {})
    secret = admin_rec.get("secret")

    import_base_ok = True
    try:
        import pyotp
        import qrcode
        from PIL import Image
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
            st.session_state.admin_unlocked = False
            st.info("Wylogowano.")
            st.rerun()

    # If no secret yet -> allow initial creation (only local)
    if not secret:
        st.warning("Brak skonfigurowanego TOTP. Utwórz sekret i dodaj go do aplikacji Authenticator na telefonie.")
        if st.button("Utwórz sekret TOTP teraz"):
            new_secret = pyotp.random_base32()
            db = _load_users()
            db["_admin_totp"] = {"secret": new_secret}
            _save_users(db)
            st.success("Sekret wygenerowany. Dodaj go do Authenticator (pokażę QR i secret).")
            st.rerun()
        st.stop()

    # Show login form (enter 6-digit code from phone)
    st.markdown("**Zaloguj się kodem z aplikacji Authenticator**")
    col_a, col_b = st.columns([2,1])
    with col_a:
        code = st.text_input("6-cyfrowy kod TOTP", max_chars=6, key="admin_code_input")
    with col_b:
        if st.button("Zaloguj administratora"):
            try:
                totp = pyotp.TOTP(secret)
                ok = totp.verify(code, valid_window=1)
                if ok:
                    st.session_state.admin_unlocked = True
                    st.success("Zalogowano jako Administrator.")
                    st.rerun()
                else:
                    st.error("Kod niepoprawny. Sprawdź w aplikacji Authenticator i spróbuj ponownie.")
            except Exception as e:
                st.error(f"Błąd weryfikacji: {e}")

    st.divider()
    st.markdown("### Konfiguracja sekretu (tylko lokalnie)")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Jeżeli chcesz skonfigurować ręcznie w aplikacji Authenticator, użyj poniższego secretu.")
        st.code(secret, language="text")
    with col2:
        if st.button("Pokaż QR (provisioning URI)"):
            try:
                totp = pyotp.TOTP(secret)
                uri = totp.provisioning_uri(name=f"{APP_NAME}-admin", issuer_name=APP_NAME)
                qr = qrcode.make(uri)
                buf = io.BytesIO()
                qr.save(buf, format="PNG")
                buf.seek(0)
                st.image(buf, caption="Zeskanuj ten QR kod w aplikacji Authenticator")
            except Exception as e:
                st.error(f"Nie udało się wygenerować QR: {e}")

    st.markdown("---")

    # if admin unlocked -> show admin controls
    if st.session_state.get("admin_unlocked", False):
        st.markdown("## 🔧 Panel administratora — operacje")
        db = _load_users()

        st.subheader("Konta użytkowników")
        if not db or all(k.startswith("_") for k in db.keys()):
            st.caption("Brak użytkowników w users.json")
        else:
            cols = st.columns([2,1,1])
            cols[0].markdown("**Login**")
            cols[1].markdown("**XP**")
            cols[2].markdown("**Akcje**")
            # show all real users (exclude internal keys starting with _)
            users_list = [k for k in db.keys() if not k.startswith("_")]
            for u in users_list:
                prof = db.get(u, {})
                xp = prof.get("xp", 0)
                c1, c2, c3 = st.columns([2,1,1])
                c1.write(u)
                c2.write(xp)
                if c3.button(f"Usuń konto: {u}", key=f"del_user_{u}"):
                    del db[u]
                    _save_users(db)
                    st.success(f"Usunięto konto: {u}")
                    st.rerun()

        st.divider()

        st.subheader("Pliki konfiguracji i backupy")
        # download users.json
        if st.button("Pobierz backup users.json"):
            try:
                st.download_button("Kliknij aby pobrać users.json", data=json.dumps(db, ensure_ascii=False, indent=2).encode("utf-8"),
                                   file_name="users_backup.json", mime="application/json")
            except Exception as e:
                st.error(f"Błąd: {e}")

        # upload new tasks.json (replace)
        st.markdown("**Zastąp plik data/tasks.json (upload)**")
        uploaded_tasks = st.file_uploader("Wgraj tasks.json (zastąpi obecny)", type=["json"], key="admin_upload_tasks")
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
                st.download_button("Pobierz tasks.json", data=content.encode("utf-8"), file_name="tasks.json", mime="application/json")
            else:
                st.info("Brak pliku data/tasks.json")

                st.divider()

        st.subheader("🎁 Konkursy i losowanie nagród")

        donors = _load_donors()
        draws = _load_draws()

        st.caption(f"Zgłoszeń konkursowych w donors.json: {len(donors)}")

        if not donors:
            st.info("Brak zgłoszeń w data/donors.json – najpierw niech rodzice wypełnią formularz w zakładce 'Wsparcie & konkursy'.")
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
                # przygotowanie puli
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

                    # zapis do historii losowań
                    draw_record = {
                        "timestamp": datetime.now(tz=tz.gettz("Europe/Warsaw")).isoformat(),
                        "mode": mode,
                        "num_candidates": len(pool),
                        "num_winners": k,
                        "winners": winners,
                    }
                    draws.append(draw_record)
                    _save_draws(draws)
                    st.info("Zapisano wynik losowania do data/draws.json")

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
        admin_action = st.radio("Akcja", ["Pokaż rekord PIN rodzica", "Resetuj PIN rodzica"], index=0)
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
            db["_admin_totp"] = {"secret": new_secret}
            _save_users(db)
            st.success("Wygenerowano nowy sekret. Zeskanuj nowy QR lub użyj secretu wyżej.")
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
        st.text_input("PIN (domyślnie 1234)", type="password", key="parent_pin_input", on_change=_try_unlock_parent)
        st.info("Wpisz PIN i naciśnij Enter.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Raport", "Dane i prywatność", "Ustawienia PIN"])

    with tab1:
        st.subheader("Raport aktywności")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Poziom", current_level(st.session_state.xp))
        c2.metric("XP", st.session_state.xp)
        c3.metric("Odznaki", len(st.session_state.badges))
        c4.metric("Naklejki", len(st.session_state.stickers))

        events = st.session_state.activity_log[-10:][::-1]
        if events:
            st.markdown("#### Ostatnie działania")
            for e in events:
                st.write(f"• {e['time']} — {e['event']}")
        else:
            st.caption("Brak zdarzeń — zacznij od strony Start lub Misje.")

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
                "dataset": st.session_state.dataset_name,
                "user": st.session_state.get("user"),
            }
            st.json(overview)
            st.download_button(
                "Pobierz raport JSON",
                data=json.dumps(overview, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="data4kids_raport.json",
                mime="application/json",
            )

    with tab2:
        st.subheader("Wgraj/usuń dane")

        if st.button("Przywróć dane przykładowe"):
            group = st.session_state.age_group
            presets = DATASETS_PRESETS[group]
            first_name = list(presets.keys())[0]
            st.session_state.data = make_dataset(120, presets[first_name], seed=random.randint(0, 9999))
            st.success("Przywrócono przykładowe dane.")

        st.divider()
        st.subheader("Prywatność (MVP)")
        st.caption("Wersja MVP nie wysyła nic w internet. Wszystko dzieje się lokalnie w przeglądarce.")

        st.divider()
        st.subheader("Usuwanie danych profilu dziecka")
        if not st.session_state.get("user"):
            st.info("Zaloguj dziecko na stronie Start, aby zarządzać jego profilem.")
        else:
            u = st.session_state.user
            st.markdown(f"Zarządzasz profilem: **{u}**. Operacja jest nieodwracalne i dotyczy wyłącznie tego profilu.")
            confirm_text = st.text_input("Aby potwierdzić, wpisz dokładnie login dziecka:",
                                         placeholder="wpisz login...", key="delete_confirm_login")
            confirm_box = st.checkbox("Rozumiem, że tej operacji nie da się cofnąć.", key="delete_confirm_check")
            if st.button("Usuń ten profil", type="secondary"):
                if not confirm_box:
                    st.warning("Zaznacz potwierdzenie, że rozumiesz konsekwencje.")
                elif confirm_text != u:
                    st.error("Login nie zgadza się. Wpisz dokładnie nazwę profilu.")
                else:
                    db = _load_users()
                    if u in db:
                        del db[u]
                        _save_users(db)
                    st.session_state.user = None
                    st.session_state.xp = 0
                    st.session_state.badges = set()
                    st.session_state.stickers = set()
                    st.success("Profil usunięty lokalnie.")

    with tab3:
        st.subheader("🔐 Zmień PIN rodzica")
        with st.form("change_parent_pin"):
            cur = st.text_input("Obecny PIN", type="password")
            new1 = st.text_input("Nowy PIN (min. 4 cyfry)", type="password")
            new2 = st.text_input("Powtórz nowy PIN", type="password")
            submitted = st.form_submit_button("Zmień PIN")
            if submitted:
                if not verify_parent_pin(cur):
                    st.error("Obecny PIN jest nieprawidłowy.")
                elif new1 != new2:
                    st.error("Nowe PINy nie są takie same.")
                else:
                    try:
                        set_parent_pin(new1)
                        st.success("PIN został zmieniony i zaczyna działać od razu.")
                    except ValueError as e:
                        st.error(str(e))

        if st.button("🔒 Zablokuj panel rodzica"):
            st.session_state["parent_unlocked"] = False
            st.info("Panel zablokowany.")
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
