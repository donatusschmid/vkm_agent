import os
import psycopg
from dotenv import load_dotenv

# ============================================================
# Konfiguration
# ============================================================

load_dotenv()
PG_DSN = os.getenv("PG_DSN", "postgresql://donatusschmid@localhost:5432/vkm_new")

def analyze_database_size():
    """Analysiert die physische Größe der Datenbank und Tabellen."""
    
    # SQL für die Tabellen-Details
    # pg_table_size: Nur die Daten der Tabelle
    # pg_indexes_size: Alle Indizes (HNSW, GIN, B-Tree)
    # pg_total_relation_size: Daten + Indizes
    query_tables = """
        SELECT 
            relname AS table_name, 
            pg_size_pretty(pg_table_size(relid)) AS data_size,
            pg_size_pretty(pg_indexes_size(relid)) AS index_size,
            pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
            pg_total_relation_size(relid) AS raw_size
        FROM pg_catalog.pg_statio_user_tables
        ORDER BY raw_size DESC;
    """
    
    query_db_total = "SELECT pg_size_pretty(pg_database_size(current_database()));"

    try:
        with psycopg.connect(PG_DSN) as conn:
            with conn.cursor() as cur:
                # 1. Gesamtgröße der DB abrufen
                cur.execute(query_db_total)
                db_total = cur.fetchone()[0]
                
                print("\n" + "="*80)
                print(f"📊 DATENBANK-ANALYSE: {PG_DSN.split('/')[-1].upper()}")
                print(f"Gesamtgröße auf Festplatte: {db_total}")
                print("="*80)
                
                # 2. Tabellen-Details abrufen
                print(f"{'Tabelle':<25} | {'Daten':<12} | {'Indizes':<12} | {'Gesamt':<12}")
                print("-" * 80)
                
                cur.execute(query_tables)
                rows = cur.fetchall()
                
                if not rows:
                    print("Keine Benutzertabellen gefunden.")
                else:
                    for row in rows:
                        print(f"{row[0]:<25} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12}")
                
                print("="*80 + "\n")

    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")

if __name__ == "__main__":
    analyze_database_size()
