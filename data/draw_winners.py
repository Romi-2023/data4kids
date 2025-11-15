import json
import os
import random
from datetime import datetime
from dateutil import tz

DATA_DIR = "data"
DONORS_FILE = os.path.join(DATA_DIR, "donors.json")
DRAWS_FILE = os.path.join(DATA_DIR, "draws.json")

def load_json_list(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_json_list(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def main():
    donors = load_json_list(DONORS_FILE)
    if not donors:
        print("Brak zgłoszeń w data/donors.json")
        return

    print(f"Liczba zgłoszeń: {len(donors)}")
    mode = input("Tryb (1 = każde zgłoszenie, 2 = unikalny e-mail) [1/2]: ").strip() or "1"
    try:
        n = int(input("Ilu zwycięzców wylosować?: ").strip())
    except ValueError:
        print("Zła liczba, przerywam.")
        return

    pool = donors
    if mode == "2":
        uniq = {}
        for d in donors:
            key = d.get("contact") or ""
            if key and key not in uniq:
                uniq[key] = d
        pool = list(uniq.values())

    if not pool:
        print("Brak prawidłowych zgłoszeń do losowania.")
        return

    n = min(n, len(pool))
    winners = random.sample(pool, k=n)

    print("\n=== ZWYCIĘZCY ===")
    for i, w in enumerate(winners, start=1):
        print(
            f"{i}. {w.get('parent_name','?')} <{w.get('contact','?')}> "
            f"(login dziecka: {w.get('child_login','-')}, kwota: {w.get('amount','?')})"
        )

    draws = load_json_list(DRAWS_FILE)
    draws.append(
        {
            "timestamp": datetime.now(tz=tz.gettz("Europe/Warsaw")).isoformat(),
            "mode": "unikalny email" if mode == "2" else "każde zgłoszenie",
            "num_candidates": len(pool),
            "num_winners": n,
            "winners": winners,
        }
    )
    save_json_list(DRAWS_FILE, draws)
    print(f"\nZapisano wynik losowania do {DRAWS_FILE}")

if __name__ == "__main__":
    main()
