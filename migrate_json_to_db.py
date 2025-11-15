import os
import json
import psycopg2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

USERS_FILE = os.path.join(DATA_DIR, "users.json")
DONORS_FILE = os.path.join(DATA_DIR, "donors.json")
DRAWS_FILE = os.path.join(DATA_DIR, "draws.json")

DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("Brak zmiennej środowiskowej DATABASE_URL")
    return psycopg2.connect(DATABASE_URL)


def ensure_kv_table():
    conn = get_db_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kv_store (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    """
                )
    finally:
        conn.close()


def kv_set_json(key: str, value) -> None:
    payload = json.dumps(value, ensure_ascii=False)
    conn = get_db_connection()
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


def load_json_if_exists(path, default):
    if not os.path.exists(path):
        print(f"[INFO] Plik nie istnieje, pomijam: {path}")
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] Wczytano dane z {path}")
        return data
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać {path}: {e}")
        return default


def main():
    print("=== Migracja danych JSON -> PostgreSQL (kv_store) ===")

    if not DATABASE_URL:
        print("❌ Brak zmiennej środowiskowej DATABASE_URL")
        print("   Ustaw ją na connection string bazy z DigitalOcean i spróbuj ponownie.")
        return

    # 1. Upewnij się, że tabela istnieje
    print("[INFO] Tworzę (jeśli potrzeba) tabelę kv_store...")
    ensure_kv_table()
    print("[OK] Tabela kv_store gotowa.")

    # 2. Users
    users = load_json_if_exists(USERS_FILE, {})
    kv_set_json("users", users)
    print(f"[OK] Zapisano 'users' do kv_store (liczba użytkowników: {len(users) if isinstance(users, dict) else 'nie dotyczy'})")

    # 3. Donors
    donors = load_json_if_exists(DONORS_FILE, [])
    kv_set_json("donors", donors)
    print(f"[OK] Zapisano 'donors' do kv_store (rekordów: {len(donors) if isinstance(donors, list) else 'nie dotyczy'})")

    # 4. Draws
    draws = load_json_if_exists(DRAWS_FILE, [])
    kv_set_json("draws", draws)
    print(f"[OK] Zapisano 'draws' do kv_store (rekordów: {len(draws) if isinstance(draws, list) else 'nie dotyczy'})")

    print("=== Migracja zakończona ✅ ===")


if __name__ == "__main__":
    main()
