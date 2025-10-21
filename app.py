# Data4Kids — Streamlit MVP (rozszerzone)
# (misje, daily quest, symulacje, czyszczenie, fabuła,
#  album naklejek, quiz obrazkowy, + NOWE: przedmioty szkolne)
# -------------------------------------------------------------------------------------------------
# Windows (PowerShell):
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   python -m pip install --upgrade pip
#   pip install -r requirements.txt
#   streamlit run app.py
# -------------------------------------------------------------------------------------------------

import json
import hashlib
from datetime import datetime, date
from dateutil import tz
import random
from typing import Optional, List, Dict

import pandas as pd
import altair as alt
import streamlit as st

APP_NAME = "Data4Kids"
VERSION = "0.8.0"

# ---------------------------------
# Utilities & basic security (MVP)
# ---------------------------------

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Parent PIN (non-persistent MVP)
PARENT_PIN_HASH = hash_text("1234")
if "pin_hash" not in st.session_state:
    st.session_state.pin_hash = PARENT_PIN_HASH

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
    if xp >= 100:
        return 4
    if xp >= 60:
        return 3
    if xp >= 30:
        return 2
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

# Presets per age group (simpler → fewer columns)
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

# -----------------------------
# Stickers catalog (rozszerzone)
# -----------------------------
STICKERS: Dict[str, Dict[str, str]] = {
    "sticker_bars": {"emoji": "📊", "label": "Mistrz Słupków", "desc": "Poprawny wykres słupkowy."},
    "sticker_points": {"emoji": "🔵", "label": "Mistrz Punktów", "desc": "Poprawny wykres punktowy."},
    "sticker_detect": {"emoji": "🍉", "label": "Arbuzowy Tropiciel", "desc": "Zadanie detektywistyczne z arbuzem."},
    "sticker_sim": {"emoji": "🎲", "label": "Badacz Symulacji", "desc": "Symulacja rzutu monetą."},
    "sticker_clean": {"emoji": "🩺", "label": "Doktor Danych", "desc": "Naprawianie literówek."},
    "sticker_story": {"emoji": "📖", "label": "Opowieściopisarz", "desc": "Fabuła piknikowa."},
    "sticker_hawkeye": {"emoji": "👁️", "label": "Oko Sokoła", "desc": "Quiz obrazkowy — spostrzegawczość."},
    # nowe — przedmioty
    "sticker_math": {"emoji": "➗", "label": "Mat-fun", "desc": "Zadanie z matematyki wykonane!"},
    "sticker_polish": {"emoji": "📝", "label": "Językowa Iskra", "desc": "Polski — części mowy/ortografia."},
    "sticker_history": {"emoji": "🏺", "label": "Kronikarz", "desc": "Historia — oś czasu."},
    "sticker_geo": {"emoji": "🗺️", "label": "Mały Geograf", "desc": "Geografia — stolice i kontynenty."},
    "sticker_physics": {"emoji": "⚙️", "label": "Fiz-Mistrz", "desc": "Fizyka — prędkość = s/t."},
    "sticker_chem": {"emoji": "🧪", "label": "Chemik Amator", "desc": "Chemia — masa molowa."},
    "sticker_english": {"emoji": "🇬🇧", "label": "Word Wizard", "desc": "Angielski — słówka/irregulars."},
    "sticker_bio": {"emoji": "🧬", "label": "Mały Biolog", "desc": "Biologia — podstawy komórki i łańcucha pokarmowego."},

}

# -----------------------------
# Session state
# -----------------------------
if "parent_unlocked" not in st.session_state:
    st.session_state.parent_unlocked = False
if "kid_name" not in st.session_state:
    st.session_state.kid_name = ""
if "age" not in st.session_state:
    st.session_state.age = None
if "age_group" not in st.session_state:
    st.session_state.age_group = "10-12"
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "data" not in st.session_state:
    st.session_state.data = make_dataset(140, DATASETS_PRESETS["10-12"]["Średni"], seed=42)
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "xp" not in st.session_state:
    st.session_state.xp = 0
if "badges" not in st.session_state:
    st.session_state.badges = set()
if "stickers" not in st.session_state:
    st.session_state.stickers = set()
if "missions_state" not in st.session_state:
    st.session_state.missions_state = {}
if "hall_of_fame" not in st.session_state:
    st.session_state.hall_of_fame = []
# Daily quest state
if "last_quest" not in st.session_state:
    st.session_state.last_quest = None
if "todays" not in st.session_state:
    st.session_state.todays = None
if "kids_mode" not in st.session_state:
    st.session_state.kids_mode = True

def log_event(event: str):
    stamp = datetime.now(tz=tz.gettz("Europe/Warsaw")).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.append({"time": stamp, "event": event})

# -----------------------------
# Glossary
# -----------------------------
GLOSSARY = {
    "średnia": "Suma wszystkich wartości podzielona przez ich liczbę.",
    "mediana": "Wartość środkowa po ułożeniu danych od najmniejszej do największej.",
    "korelacja": "Miara tego, jak dwie rzeczy zmieniają się razem (dodatnia, ujemna, brak).",
    "agregacja": "Łączenie danych (np. liczenie średniej) w grupach.",
    "kategoria": "Słowo/etykieta zamiast liczby (np. kolor, miasto).",
}

# Specjalna etykieta przyjazna dzieciom dla count()
COUNT_LABEL = "liczba osób"

def _is_count_choice(val: str) -> bool:
    return val in ("count()", COUNT_LABEL)

# -----------------------------
# Global helpers (do użycia w różnych stronach)
# -----------------------------

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

def grant_sticker(code: str):
    if code in STICKERS:
        st.session_state.stickers.add(code)

def show_hint(mid: str, hint: str):
    key = f"hint_used_{mid}"
    if st.button("Podpowiedź 🪄 (-1 XP)", key=f"hintbtn_{mid}"):
        if not st.session_state.get(key, False):
            st.session_state.xp = max(0, st.session_state.xp - 1)
            st.session_state[key] = True
        st.caption(hint)

# -----------------------------
# Chemistry constants + parser
# -----------------------------
ATOMIC_MASS = {"H": 1.008, "C": 12.011, "O": 15.999, "N": 14.007, "Na": 22.990, "Cl": 35.45}

def _molar_mass(formula: str) -> Optional[float]:
    # prosty parser: obsługa H2O, CO2, NaCl, C6H12O6 itp. (bez nawiasów)
    import re
    tokens = re.findall(r"[A-Z][a-z]?\d*", formula)
    if not tokens:
        return None
    total = 0.0
    for tok in tokens:
        m = re.match(r"([A-Z][a-z]?)(\d*)", tok)
        if not m:
            return None
        el, num = m.group(1), m.group(2)
        if el not in ATOMIC_MASS:
            return None
        n = int(num) if num else 1
        total += ATOMIC_MASS[el] * n
    return total

# -----------------------------
# Missions (global definitions)
# -----------------------------
def mission_draw_xy(mid: str, req_x: str, req_y: str, req_type: str) -> None:
    display_req_y = COUNT_LABEL if _is_count_choice(req_y) else req_y
    st.write(f"**Zadanie:** Narysuj wykres: **{req_type}** z osią **X={req_x}**, **Y={display_req_y}**.")
    df = st.session_state.data

    x = st.selectbox("Oś X", df.columns.tolist(), key=f"{mid}_x")
    y_options = [COUNT_LABEL] + df.columns.tolist()
    y = st.selectbox("Oś Y", y_options, key=f"{mid}_y")
    chart_type = st.selectbox("Typ wykresu", ["punktowy", "słupkowy"], key=f"{mid}_type")

    try:
        if chart_type == "punktowy":
            if _is_count_choice(y):
                st.warning("Dla wykresu punktowego wybierz kolumnę liczbową na osi Y (nie 'liczba osób').")
                ch = alt.Chart(df).mark_circle(size=70, opacity=0.7).encode(x=x, y=x, tooltip=[x])
            else:
                ch = alt.Chart(df).mark_circle(size=70, opacity=0.7).encode(x=x, y=y, tooltip=[x, y])
        else:  # słupkowy
            if _is_count_choice(y):
                ch = alt.Chart(df).mark_bar().encode(x=x, y=alt.Y("count():Q", title="Liczba osób"), tooltip=[x])
            else:
                ch = alt.Chart(df).mark_bar().encode(x=x, y=y, tooltip=[x, y])
        st.altair_chart(ch.interactive(), use_container_width=True)
    except Exception as e:
        st.warning(f"Nie udało się narysować: {e}")

    y_ok = (_is_count_choice(y) and _is_count_choice(req_y)) or (y == req_y)
    ok = (x == req_x) and y_ok and (chart_type == req_type)

    if st.button(f"Sprawdź {mid}"):
        award(ok, 10, badge="Rysownik danych", mid=mid)
        if ok:
            grant_sticker("sticker_bars" if chart_type == "słupkowy" else "sticker_points")
            st.success("✅ Super — dokładnie taki wykres!")
        else:
            st.warning(f"Jeszcze nie. Ustaw X={req_x}, Y={display_req_y}, typ={req_type}.")
    show_hint(mid, "Słupki liczą **liczbę osób**, a punkty wymagają liczb na osi Y.")

def mission_detect_city(mid: str) -> None:
    st.write("**Zadanie detektywistyczne:** Znajdź **miasto**, w którym jest **co najmniej 5 osób** i ich **ulubiony owoc to 'arbuz'**.")

    df = st.session_state.data

    required = {"miasto", "ulubiony_owoc"}
    if not required.issubset(set(df.columns)):
        st.error("Potrzebne kolumny: 'miasto', 'ulubiony_owoc'.")
        return

    # Normalizacja + filtr na arbuz (case-insensitive)
    norm = df.copy()
    norm["miasto"] = norm["miasto"].astype(str).str.strip()
    norm["ulubiony_owoc"] = norm["ulubiony_owoc"].astype(str).str.strip().str.lower()
    df_arbuz = norm[norm["ulubiony_owoc"] == "arbuz"]

    grp = (
        df_arbuz
        .groupby("miasto", as_index=False)
        .size()
        .rename(columns={"size": "liczba_osób"})
        .sort_values("liczba_osób", ascending=False)
    )

    st.write("Zobacz wartości w tabeli lub narysuj słupki: X=miasto, Y=liczba osób (arbuz).")

    if grp.empty:
        st.info("Brak danych o fanach arbuza 🍉.")
        return

    st.dataframe(grp, use_container_width=True)
    st.bar_chart(grp.set_index("miasto")["liczba_osób"])

    city_pick = st.selectbox("Twoje miasto:", grp["miasto"].tolist(), key=f"{mid}_city")

    if st.button(f"Sprawdź {mid}", key=f"{mid}_check"):
        liczba = int(grp.loc[grp["miasto"] == city_pick, "liczba_osób"].iloc[0]) if city_pick in grp["miasto"].values else 0
        ok = liczba >= 5
        award(ok, 15, badge="Sherlock danych", mid=mid)
        if ok:
            grant_sticker("sticker_detect")
            st.success("✅ Brawo! To miasto spełnia warunek (≥ 5 osób z arbuzem).")
        else:
            st.warning(f"W {city_pick} jest tylko {liczba} fanów arbuza. Poszukaj miasta z wynikiem ≥ 5.")

        show_hint(mid, "Przefiltruj na 'arbuz', zgrupuj po mieście, policz i wybierz miasto z wynikiem ≥ 5.")

def mission_fill_blank_text(mid: str, sentence_tpl: str, correct_word: str, options: List[str], xp_gain: int = 6) -> None:
    st.write("**Uzupełnij zdanie:**")
    st.write(sentence_tpl.replace("___", "**___**"))
    pick = st.selectbox("Wybierz słowo:", options, key=f"{mid}_pick")

    if st.button(f"Sprawdź {mid}"):
        ok = pick == correct_word
        award(ok, xp_gain, badge="Mistrz słówek", mid=mid)
        if ok:
            st.success("✅ Dobrze!")
        else:
            st.warning(f"Jeszcze nie. Poprawna odpowiedź: **{correct_word}**")

    show_hint(mid, "Na osi Y w słupkach często jest **liczba osób**.")

def mission_fill_number(mid: str, prompt: str, true_value: float, tolerance: Optional[float] = None, xp_gain: int = 8) -> None:
    st.write(f"**Uzupełnij liczbę:** {prompt}")
    step = 0.1 if isinstance(true_value, float) and not float(true_value).is_integer() else 1
    guess = st.number_input("Twoja odpowiedź:", step=step, key=f"{mid}_num")

    if st.button(f"Sprawdź {mid}"):
        ok = (abs(guess - true_value) <= tolerance) if tolerance is not None else (guess == true_value)
        award(ok, xp_gain, badge="Liczydło", mid=mid)
        if ok:
            st.success(f"✅ Tak! Prawidłowo: {true_value:g}.")
        else:
            st.warning(f"Prawidłowo: {true_value:g}.")

    show_hint(mid, "Policz średnią: dodaj wszystkie i podziel przez liczbę osób.")

def mission_order_steps(mid: str, prompt: str, steps_correct: List[str], xp_gain: int = 10) -> None:
    st.write(f"**Ułóż w kolejności:** {prompt}")
    picked = st.multiselect("Klikaj kroki we właściwej kolejności ⬇️", steps_correct, default=[], key=f"{mid}_order")
    st.caption("Tip: klikaj po kolei; lista u góry zachowuje kolejność wyboru.")

    if st.button(f"Sprawdź {mid}"):
        ok = picked == steps_correct
        award(ok, xp_gain, badge="Porządny planista", mid=mid)
        if ok:
            st.success("✅ Idealnie ułożone!")
        else:
            st.warning("Jeszcze nie. Zacznij od **Wczytaj dane** i skończ na **Zapisz wynik**.")

    show_hint(mid, "Najpierw **wczytaj**, potem **wybierz kolumny**, potem **wykres**.")

def mission_spot_the_error(mid: str, df_local: pd.DataFrame, xp_gain: int = 12) -> None:
    st.write("**Znajdź błąd na wykresie:**")
    if all(c in df_local.columns for c in ["ulubiony_owoc", "wiek"]):
        bad = alt.Chart(df_local).mark_bar().encode(x="ulubiony_owoc:N", y="wiek:Q", tooltip=["ulubiony_owoc", "wiek"])
        st.altair_chart(bad, use_container_width=True)
        q = "Co jest nie tak?"
        opts = [
            "Na osi Y powinna być 'liczba osób', nie 'wiek'.",
            "Na osi X powinna być liczba, nie kategoria.",
            "Kolory są złe.",
        ]
        pick = st.radio(q, opts, index=None, key=f"{mid}_err")
        if st.button(f"Sprawdź {mid}"):
            ok = pick == opts[0]
            award(ok, xp_gain, badge="Detektyw wykresów", mid=mid)
            if ok:
                st.success("✅ Dokładnie!")
            else:
                st.warning("Spróbuj jeszcze raz; pomyśl o tym, co liczą słupki.")
    else:
        st.info("Załaduj zestaw z kolumnami 'ulubiony_owoc' i 'wiek'.")

    show_hint(mid, "Słupki zwykle liczą, ile elementów jest w każdej kategorii.")

def mission_simulate_coin(mid: str) -> None:
    st.write("**Symulacja rzutu monetą 🎲** — wybierz liczbę rzutów, zgadnij udział orłów, potem sprawdź!")
    n = st.selectbox("Liczba rzutów:", [10, 100, 1000], index=1, key=f"{mid}_n")
    guess = st.slider("Twoja zgadywana proporcja orłów", 0.0, 1.0, 0.5, 0.01, key=f"{mid}_g")
    tol = 0.10 if n == 10 else (0.05 if n == 100 else 0.03)

    if st.button(f"Symuluj {mid}"):
        flips = [random.choice(["orzeł", "reszka"]) for _ in range(n)]
        heads = flips.count("orzeł")
        prop = heads / n
        st.write(f"Wynik: orły = {heads}/{n} (≈ {prop:.2f})")
        df_sim = pd.DataFrame({"wynik": flips})
        chart = alt.Chart(df_sim).mark_bar().encode(x="wynik:N", y=alt.Y("count():Q", title="Liczba"))
        st.altair_chart(chart, use_container_width=True)
        ok = abs(prop - guess) <= tol
        award(ok, 10, badge="Mały probabilista", mid=mid)
        if ok:
            grant_sticker("sticker_sim")
            st.success("✅ Świetna estymacja!")
        else:
            st.info("Nie szkodzi! Im więcej rzutów, tym bliżej 0.5.")

    show_hint(mid, "Przy dużej liczbie rzutów wynik zbliża się do 50% orłów.")

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

def mission_history_timeline(mid: str):
    st.subheader("Historia 🏺: ułóż oś czasu")
    events = [
        ("Chrzest Polski", 966),
        ("Bitwa pod Grunwaldem", 1410),
        ("Konstytucja 3 Maja", 1791),
        ("Odzyskanie niepodległości", 1918),
    ]
    labels = [e[0] for e in events]
    order = st.multiselect("Klikaj w kolejności od najstarszego do najmłodszego", labels, key=f"{mid}_ord")
    if st.button(f"Sprawdź {mid}"):
        correct = [e[0] for e in sorted(events, key=lambda x: x[1])]
        ok = (order == correct)
        award(ok, 9, badge="Kronikarz", mid=mid)
        if ok:
            grant_sticker("sticker_history")
            st.success("✅ Pięknie ułożone!")
        else:
            st.warning("Podpowiedź: 966 → 1410 → 1791 → 1918")

def mission_geo_capitals(mid: str):
    st.subheader("Geografia 🗺️: stolice")
    pairs = {
        "Polska": "Warszawa",
        "Niemcy": "Berlin",
        "Francja": "Paryż",
        "Hiszpania": "Madryt",
    }

    # —— utrwal losowanie na czas rozwiązywania zadania ——
    state_key = f"{mid}_country"
    if state_key not in st.session_state:
        st.session_state[state_key] = random.choice(list(pairs.keys()))
    country = st.session_state[state_key]

    # opcjonalny przycisk: losuj kolejne pytanie
    if st.button("Wylosuj inne państwo", key=f"{mid}_new"):
        st.session_state[state_key] = random.choice(list(pairs.keys()))
        st.rerun()

    pick = st.selectbox(
        f"Stolica kraju: {country}",
        ["Warszawa", "Berlin", "Paryż", "Madryt"],
        key=f"{mid}_pick",
    )

    if st.button(f"Sprawdź {mid}"):
        ok = (pick == pairs[country])
        award(ok, 7, badge="Mały Geograf", mid=mid)
        if ok:
            grant_sticker("sticker_geo")
            st.success("✅ Super!")
        else:
            st.warning(f"Prawidłowo: {pairs[country]}")


def mission_physics_speed(mid: str):
    st.subheader("Fizyka ⚙️: prędkość = droga / czas")
    s = random.choice([100, 150, 200, 240])  # metry
    t = random.choice([5, 8, 10, 12])        # sekundy
    guess = st.number_input(f"Oblicz prędkość dla s={s} m, t={t} s (m/s)", step=1.0, key=f"{mid}_v")
    true = s / t
    if st.button(f"Sprawdź {mid}"):
        ok = abs(guess - true) <= 0.1
        award(ok, 8, badge="Fiz-Mistrz", mid=mid)
        if ok:
            grant_sticker("sticker_physics")
            st.success("✅ Git!")
        else:
            st.warning(f"Prawidłowo ≈ {true:.2f} m/s")
    show_hint(mid, "Wzór: v = s / t. Uważaj na jednostki!")

def mission_chem_molar(mid: str):
    st.subheader("Chemia 🧪: masa molowa")
    choices = ["H2O", "CO2", "NaCl", "C6H12O6"]
    pick = st.selectbox("Wybierz wzór:", choices, key=f"{mid}_f")
    guess = st.number_input("Podaj masę molową (g/mol)", step=0.1, key=f"{mid}_m")
    if st.button(f"Sprawdź {mid}"):
        mm = _molar_mass(pick)
        if mm is None:
            st.warning("Nieobsługiwany wzór.")
            return
        ok = abs(guess - mm) <= 1.0
        award(ok, 10, badge="Chemik Amator", mid=mid)
        if ok:
            grant_sticker("sticker_chem")
            st.success("✅ Dobrze!")
        else:
            st.warning(f"Wynik ≈ {mm:.2f} g/mol")
    show_hint(mid, "Zsumuj masy atomowe pierwiastków pomnożone przez indeksy.")

def mission_english_irregular(mid: str):
    st.subheader("Angielski 🇬🇧: irregular verbs")
    verbs = {"go": "went", "see": "saw", "eat": "ate", "have": "had", "make": "made"}
    base = random.choice(list(verbs.keys()))
    pick = st.selectbox(f"Past Simple od '{base}' to…", sorted(set(verbs.values()) | {"goed", "seed"}), key=f"{mid}_v")
    if st.button(f"Sprawdź {mid}"):
        ok = (pick == verbs[base])
        award(ok, 7, badge="Word Wizard", mid=mid)
        if ok:
            grant_sticker("sticker_english")
            st.success("✅ Nice!")
        else:
            st.warning(f"Prawidłowo: {verbs[base]}")

def mission_bio_mito(mid: str):
    st.subheader("Biologia 🧬: Komórka – co robi mitochondrium?")
    question = "Który element komórki odpowiada za produkcję energii?"
    options = ["jądro komórkowe", "mitochondrium", "błona komórkowa", "chloroplast"]
    pick = st.radio(question, options, index=None, key=f"{mid}_pick")

    if st.button(f"Sprawdź {mid}"):
        ok = (pick == "mitochondrium")
        award(ok, 7, badge="Mały Biolog", mid=mid)
        if ok:
            grant_sticker("sticker_bio")
            st.success("✅ Tak! Mitochondrium to „elektrownia” komórki.")
        else:
            st.warning("To nie to. Podpowiedź: „elektrownia” komórki = mitochondrium.")
    show_hint(mid, "Mitochondria wytwarzają ATP — paliwo energetyczne komórki.")

def mission_bio_foodchain(mid: str):
    st.subheader("Biologia 🧬: Łańcuch pokarmowy – kto jest kim?")
    bank = [
        {"prompt": "Kto jest producentem?", "options": ["trawa", "zając", "wilk"], "answer": "trawa"},
        {"prompt": "Kto jest konsumentem I rzędu?", "options": ["zając", "trawa", "słońce"], "answer": "zając"},
        {"prompt": "Kto jest drapieżnikiem (konsument wyższego rzędu)?", "options": ["wilk", "trawa", "zając"], "answer": "wilk"},
    ]
    key_q = f"{mid}_qidx"
    if key_q not in st.session_state:
        st.session_state[key_q] = random.randrange(len(bank))
    q = bank[st.session_state[key_q]]

    colL, colR = st.columns([3,1])
    with colL:
        pick = st.radio(q["prompt"], q["options"], index=None, key=f"{mid}_pick")
    with colR:
        if st.button("Wylosuj inne pytanie", key=f"{mid}_new"):
            st.session_state[key_q] = random.randrange(len(bank))
            st.rerun()

    if st.button(f"Sprawdź {mid}", key=f"{mid}_check"):
        ok = (pick == q["answer"])
        award(ok, 8, badge="Mały Biolog", mid=mid)
        if ok:
            grant_sticker("sticker_bio")
            st.success("✅ Brawo! Poprawna odpowiedź.")
        else:
            st.warning(f"Niestety nie. Poprawna odpowiedź: **{q['answer']}**.")
    show_hint(mid, "Producent = roślina (tworzy pokarm dzięki fotosyntezie).")

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
            "Quiz danych",
            "Quiz obrazkowy",
            "Album naklejek",
            "Słowniczek",
            "Hall of Fame",
            "Panel rodzica",
        ],
    )
    # Prostszy widok dla dzieci (ukrywa JSON-y, pokazuje kafelki)
    st.checkbox("Tryb dziecięcy (prostszy widok)", key="kids_mode")

    with st.expander("Słowniczek (skrót)"):
        for k, v in GLOSSARY.items():
            st.write(f"**{k}** — {v}")

# -----------------------------
# START
# -----------------------------
if page == "Start":
    st.markdown(f"<div class='big-title'>🧒 {KID_EMOJI} Witaj w {APP_NAME}!</div>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    with colA:
        st.text_input("Twoje imię (opcjonalnie)", key="kid_name")
        age_in = st.number_input("Ile masz lat?", min_value=7, max_value=14, step=1, value=10)
        st.session_state.age = int(age_in)
        st.session_state.age_group = age_to_group(int(age_in))
        group = st.session_state.age_group
        st.info(f"Twoja grupa wiekowa: **{group}**")

        # Dataset presets by age group
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
            + "".join([f"<span class='badge'>🏅 {b}</span>" for b in st.session_state.badges]),
            unsafe_allow_html=True,
        )

# -----------------------------
# POZNAJ DANE
# -----------------------------
elif page == "Poznaj dane":
    st.markdown(f"<div class='big-title'>📊 {KID_EMOJI} Poznaj dane</div>", unsafe_allow_html=True)
    df = st.session_state.data.copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Liczba wierszy", len(df))
    if "wiek" in df.columns:
        c2.metric("Śr. wiek", round(df["wiek"].mean(), 1))
    if "wzrost_cm" in df.columns:
        c3.metric("Śr. wzrost (cm)", round(df["wzrost_cm"].mean(), 1))
    if "miasto" in df.columns:
        c4.metric("Miasta", df["miasto"].nunique())

    with st.expander("Zobacz tabelę"):
        st.dataframe(df.head(50))

# -----------------------------
# PLAC ZABAW
# -----------------------------
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

    cols = st.multiselect(
        "Kolumny do podglądu",
        st.session_state.data.columns.tolist(),
        default=st.session_state.data.columns[:4].tolist(),
    )
    st.dataframe(st.session_state.data[cols].head(30))

# -----------------------------
# MISJE — interaktywne zadania
# -----------------------------
elif page == "Misje":
    st.markdown(f"<div class='big-title'>🎯 {KID_EMOJI} Misje</div>", unsafe_allow_html=True)
    df = st.session_state.data
    xp = st.session_state.xp
    lvl = current_level(xp)

    st.info(f"Twój poziom: **L{lvl}** (progi: 30/60/100 XP) | XP: **{xp}**")

    # Daily Quest (jako funkcja lokalna korzystająca z df i lvl)
    def render_daily_quest(df: pd.DataFrame, lvl: int):
        today = date.today().isoformat()
        if st.session_state.last_quest != today or st.session_state.todays is None:
            pool = ["DQ_avg_height", "DQ_draw_bar"]
            if lvl >= 2:
                pool += ["DQ_detect"]
            if lvl >= 3:
                pool += ["DQ_sim"]
            st.session_state.todays = random.choice(pool)
            st.session_state.last_quest = today
        dq = st.session_state.todays
        st.subheader("🗓️ Dzisiejsze wyzwanie")
        if dq == "DQ_avg_height" and "wzrost_cm" in df.columns:
            avg_h = round(float(df["wzrost_cm"].mean()), 1)
            mission_fill_number("DQ-AVG", "Średni wzrost (cm) ≈", avg_h, tolerance=1.0)
        elif dq == "DQ_draw_bar" and "ulubiony_owoc" in df.columns:
            mission_draw_xy("DQ-BAR", req_x="ulubiony_owoc", req_y="count()", req_type="słupkowy")
        elif dq == "DQ_detect" and all(c in df.columns for c in ["miasto", "ulubiony_owoc"]):
            mission_detect_city("DQ-DET")
        elif dq == "DQ_sim":
            mission_simulate_coin("DQ-SIM")
        else:
            st.caption("Brakuje potrzebnych kolumn dla dzisiejszego wyzwania — zmień preset danych na Start.")

    # RENDER: Daily Quest + zestawy
    render_daily_quest(df, lvl)
    st.divider()

    with st.expander("Zestaw L1 (0+ XP) — podstawy rysowania", expanded=(lvl == 1)):
        if "wiek" in df.columns and "wzrost_cm" in df.columns:
            mission_draw_xy("L1-M1", req_x="wiek", req_y="wzrost_cm", req_type="punktowy")
        else:
            st.info("Załaduj zestaw danych z kolumnami 'wiek' i 'wzrost_cm'.")
        if "ulubiony_owoc" in df.columns:
            mission_draw_xy("L1-M2", req_x="ulubiony_owoc", req_y="count()", req_type="słupkowy")
            true_cnt = int((df["ulubiony_owoc"] == "banan").sum())
            mission_fill_blank_text("L1-M3", "Na wykresie słupkowym oś Y pokazuje ___", "liczba osób", ["liczba osób", "kolor", "imię"])
            mission_fill_number("L1-M4", "Ile osób lubi banany?", true_cnt, tolerance=None)
        else:
            st.info("Załaduj zestaw z kolumną 'ulubiony_owoc'.")

    if lvl >= 2:
        with st.expander("Zestaw L2 (30+ XP) — rysowanie + detektyw"):
            if "miasto" in df.columns and "wynik_matematyka" in df.columns:
                mission_draw_xy("L2-M1", req_x="miasto", req_y="wynik_matematyka", req_type="słupkowy")
            else:
                st.info("Potrzebne kolumny: 'miasto', 'wynik_matematyka'.")
            if "miasto" in df.columns and "ulubiony_owoc" in df.columns:
                mission_detect_city("L2-M2")
            else:
                st.info("Potrzebne kolumny: 'miasto' i 'ulubiony_owoc'.")
            mission_order_steps("L2-M3", "Ułóż kroki analizy danych od początku do końca:", ["Wczytaj dane", "Wybierz kolumny", "Narysuj wykres", "Zapisz wynik"])
            if "wzrost_cm" in df.columns:
                avg_h = round(float(df["wzrost_cm"].mean()), 1)
                mission_fill_number("L2-M4", "Średni wzrost (cm) ≈", avg_h, tolerance=1.0)

    if lvl >= 3:
        with st.expander("Zestaw L3 (60+ XP) — rozbudowane rysowanie, symulacje"):
            if all(c in df.columns for c in ["wiek", "wzrost_cm", "miasto"]):
                mission_draw_xy("L3-M1", req_x="wiek", req_y="wzrost_cm", req_type="punktowy")
            if "ulubione_zwierze" in df.columns:
                mission_draw_xy("L3-M2", req_x="ulubione_zwierze", req_y="count()", req_type="słupkowy")
            else:
                st.info("Potrzebna kolumna: 'ulubione_zwierze'.")
            mission_spot_the_error("L3-M3", df)
            mission_simulate_coin("L3-M4")

# -----------------------------
# PRZEDMIOTY SZKOLNE — NOWA STRONA
# -----------------------------
elif page == "Przedmioty szkolne":
    st.markdown(f"<div class='big-title'>📚 {KID_EMOJI} Przedmioty szkolne</div>", unsafe_allow_html=True)
    st.caption("Zadania tematyczne: matematyka, polski, historia, geografia, fizyka, chemia, angielski. Wszystko na XP i z naklejkami!")

    tab_math, tab_pol, tab_hist, tab_geo, tab_phys, tab_chem, tab_eng, tab_bio = st.tabs(
        ["Matematyka", "Język polski", "Historia", "Geografia", "Fizyka", "Chemia", "Angielski", "Biologia",])

    with tab_math:
        mission_math_arith("MAT-1")
        mission_math_line("MAT-2")

    with tab_pol:
        mission_polish_pos("POL-1")

    with tab_hist:
        mission_history_timeline("HIS-1")

    with tab_geo:
        mission_geo_capitals("GEO-1")

    with tab_phys:
        mission_physics_speed("FIZ-1")

    with tab_chem:
        mission_chem_molar("CHE-1")

    with tab_eng:
        mission_english_irregular("ANG-1")
        
    with tab_bio:
        mission_bio_mito("BIO-1")
        mission_bio_foodchain("BIO-2")

# -----------------------------
# QUIZ (tekstowy)
# -----------------------------
elif page == "Quiz danych":
    st.markdown(f"<div class='big-title'>🧠 {KID_EMOJI} Quiz danych</div>", unsafe_allow_html=True)
    group = st.session_state.age_group

    Q = {}
    Q["7-9"] = [
        ("Co to są dane?", ["Informacje o rzeczach lub osobach", "Zawsze tylko liczby", "Zagadki bez odpowiedzi"], 0),
        ("Co pokazuje wykres słupkowy?", ["Ile czegoś jest", "Kto wygra mecz", "Kolory tęczy"], 0),
        ("Co oznacza średnia?", ["Suma podzielona przez liczbę rzeczy", "Największa wartość", "Pierwsza wartość"], 0),
    ]
    Q["10-12"] = [
        ("Mediana to…", ["Wartość środkowa", "Najmniejsza wartość", "Suma wszystkiego"], 0),
        ("Punkt na wykresie punktowym to…", ["Para (X,Y)", "Zawsze liczba całkowita", "Kolor"], 0),
        ("Co robi grupowanie danych?", ["Łączy według kategorii", "Usuwa błędy", "Zwiększa liczbę wierszy"], 0),
    ]
    Q["13-14"] = [
        ("Współczynnik korelacji bliski 1 oznacza…", ["Silną zależność dodatnią", "Brak zależności", "Silną zależność ujemną"], 0),
        ("Wykres pudełkowy (boxplot) najlepiej pokazuje…", ["Rozkład i wartości odstające", "Kolory kategorii", "Kolejność dat"], 0),
        ("Średnia wrażliwa jest na…", ["Wartości odstające", "Nazwy kolumn", "Kolejność wierszy"], 0),
    ]

    questions = Q.get(group, Q["10-12"])
    answers = []
    for i, (prompt, options, correct_idx) in enumerate(questions, start=1):
        ans = st.radio(f"{i}) {prompt}", options, index=None, key=f"q_{i}")
        answers.append((ans, correct_idx, options))

    if st.button("Sprawdź odpowiedzi ✅"):
        score = 0
        for ans, correct_idx, options in answers:
            if ans is not None and ans == options[correct_idx]:
                score += 1
        if score == len(questions):
            st.success("Perfekcyjnie! 🏅 Zdobywasz odznakę: Młody Analityk!")
            st.session_state.badges.add("Młody Analityk")
            st.session_state.xp += 10
        elif score >= len(questions) - 1:
            st.info("Bardzo dobrze! Jeszcze chwilka i będzie złoto ✨")
            st.session_state.xp += 6
        else:
            st.warning("Damy radę! Wróć do 'Poznaj dane' i spróbuj ponownie.")
            st.session_state.xp += 2

# -----------------------------
# QUIZ OBRAZKOWY
# -----------------------------
elif page == "Quiz obrazkowy":
    st.markdown(f"<div class='big-title'>🖼️ {KID_EMOJI} Quiz obrazkowy</div>", unsafe_allow_html=True)

    st.subheader("Pytanie 1 — Która kategoria jest najpopularniejsza?")
    df_bar = pd.DataFrame({"owoc": ["jabłko", "banan", "truskawka", "arbuz"], "liczba": [12, 18, 9, 14]})
    chart1 = alt.Chart(df_bar).mark_bar().encode(x="owoc:N", y="liczba:Q")
    st.altair_chart(chart1, use_container_width=True)
    pick1 = st.radio("Wybierz odpowiedź:", df_bar["owoc"].tolist(), index=None, key="imgq1")

    st.subheader("Pytanie 2 — Jaki jest znak zależności?")
    rng = random.Random(123)
    xs = list(range(10))
    ys = [x + rng.randint(-2, 2) for x in xs]
    df_sc = pd.DataFrame({"x": xs, "y": ys})
    chart2 = alt.Chart(df_sc).mark_circle(size=70, opacity=0.8).encode(x="x:Q", y="y:Q")
    st.altair_chart(chart2, use_container_width=True)
    pick2 = st.radio("Wybierz odpowiedź:", ["dodatnia", "brak", "ujemna"], index=None, key="imgq2")

    if st.button("Sprawdź quiz obrazkowy ✅"):
        ok1 = (pick1 == df_bar.sort_values("liczba", ascending=False)["owoc"].iloc[0]) if pick1 is not None else False
        ok2 = (pick2 == "dodatnia") if pick2 is not None else False
        score = int(ok1) + int(ok2)
        if score == 2:
            st.success("✅ Perfekcyjnie! Dostajesz 8 XP i naklejkę 'Oko Sokoła'.")
            st.session_state.xp += 8
            st.session_state.stickers.add("sticker_hawkeye")
        elif score == 1:
            st.info("Prawie! 5 XP za wysiłek.")
            st.session_state.xp += 5
        else:
            st.warning("Spróbuj jeszcze raz — przyjrzyj się dokładnie wykresom.")
            st.session_state.xp += 2

# -----------------------------
# ALBUM NAKLEJEK
# -----------------------------
elif page == "Album naklejek":
    st.markdown("# 🗂️ Album naklejek")
    st.caption("Zbieraj naklejki, wykonując misje, quizy i eksperymenty!")

    owned = st.session_state.stickers
    total = len(STICKERS)
    st.write(f"Zebrane: **{len(owned)}/{total}**")

    for code, meta in STICKERS.items():
        owned_flag = code in owned
        css = "" if owned_flag else " locked"
        st.markdown(
            f"<div class='sticker{css}'>"
            f"<span style='font-size:1.6rem'>{meta['emoji']}</span> "
            f"<b>{meta['label']}</b> — {meta['desc']}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.download_button(
        "Pobierz mój album (JSON)",
        data=json.dumps(sorted(list(owned)), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="album_naklejek.json",
        mime="application/json",
    )

# -----------------------------
# SŁOWNICZEK
# -----------------------------
elif page == "Słowniczek":
    st.markdown("# 📖 Słowniczek pojęć")
    term = st.text_input("Szukaj pojęcia…", "")
    items = {k: v for k, v in GLOSSARY.items() if term.lower() in k.lower()}
    for k, v in items.items():
        st.write(f"**{k}** — {v}")
    if not items:
        st.caption("Brak wyników — spróbuj innego słowa.")

# -----------------------------
# HALL OF FAME & profile save
# -----------------------------
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

    # --- NOWY, przyjazny widok ---
    st.subheader("Mój profil")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imię", st.session_state.kid_name or "—")
    c2.metric("Wiek", st.session_state.age or "—")
    c3.metric("Poziom", current_level(st.session_state.xp))
    c4.metric("XP", st.session_state.xp)

    st.caption(
        f"Odznaki: **{len(st.session_state.badges)}**  |  Naklejki: **{len(st.session_state.stickers)}**"
    )

    # Plik profilu do pobrania – zostaje na wierzchu
    st.download_button(
        "Pobierz mój profil (JSON)",
        data=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="data4kids_profile.json",
        mime="application/json",
    )

    # JSON techniczny: tylko gdy wyłączony tryb dziecięcy
    if not st.session_state.get("kids_mode", True):
        with st.expander("Pokaż dane techniczne (JSON)"):
            st.json(profile)

    st.divider()
    st.subheader("Tabela Hall of Fame")
    hof_file = st.file_uploader("Wgraj istniejący hall_of_fame.json (opcjonalnie)", type=["json"])
    if hof_file is not None:
        try:
            hof_data = json.load(hof_file)
            if isinstance(hof_data, list):
                st.session_state.hall_of_fame = hof_data
                st.success("Wczytano istniejący Hall of Fame.")
            else:
                st.warning("Plik powinien zawierać listę profili (JSON array).")
        except Exception as e:
            st.error(f"Błąd wczytywania JSON: {e}")

    if st.button("Dodaj mój profil do Hall of Fame"):
        st.session_state.hall_of_fame.append(profile)
        st.success("Dodano! 🎉")

    if st.session_state.hall_of_fame:
        df_hof = pd.DataFrame(st.session_state.hall_of_fame)
        df_hof = df_hof.sort_values(by=["level", "xp"], ascending=[False, False])
        st.dataframe(df_hof)
        st.download_button(
            "Pobierz zaktualizowany hall_of_fame.json",
            data=json.dumps(st.session_state.hall_of_fame, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="hall_of_fame.json",
            mime="application/json",
        )
    else:
        st.caption("Brak wpisów — dodaj pierwszy profil!")

# -----------------------------
# PANEL RODZICA
# -----------------------------
else:
    st.markdown(f"<div class='big-title'>{PARENT_EMOJI} Panel rodzica</div>", unsafe_allow_html=True)

    if not st.session_state.parent_unlocked:
        st.markdown("Wpisz PIN, by odblokować ustawienia:")
        pin = st.text_input("PIN (domyślnie 1234)", type="password")
        if st.button("Odblokuj"):
            if hash_text(pin) == st.session_state.pin_hash:
                st.session_state.parent_unlocked = True
                log_event("parent_unlocked")
                st.success("Odblokowano panel rodzica.")
            else:
                st.error("Zły PIN.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Raport", "Dane i prywatność", "Zaawansowane (MVP)"])

    with tab1:
        st.subheader("Raport aktywności (MVP)")
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

    with tab3:
        st.subheader("Eksperymentalne")
        new_pin = st.text_input("Ustaw nowy PIN (4 cyfry)", max_chars=4)
        if st.button("Zmień PIN"):
            if new_pin and new_pin.isdigit() and len(new_pin) == 4:
                st.session_state.pin_hash = hash_text(new_pin)
                st.success("PIN zmieniony (działa od razu w tej sesji).")
            else:
                st.error("Podaj dokładnie 4 cyfry.")

        if st.button("Zablokuj panel"):
            st.session_state.parent_unlocked = False
            st.info("Panel zablokowany.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    f"<span class='muted'>v{VERSION} — {APP_NAME}. Zrobione z ❤️ w Streamlit. "
    f"<span class='pill kid'>daily quest</span> <span class='pill kid'>misje</span> <span class='pill kid'>symulacje</span> <span class='pill kid'>czyszczenie</span> <span class='pill kid'>fabuła</span> <span class='pill kid'>przedmioty</span> <span class='pill kid'>album</span> <span class='pill kid'>quizy</span> <span class='pill parent'>panel rodzica</span></span>",
    unsafe_allow_html=True,
)
