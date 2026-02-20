import os
from google import genai
from dotenv import load_dotenv

def list_gemini_models():
    # .env laden
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ Fehler: GEMINI_API_KEY nicht in der .env Datei gefunden.")
        return

    try:
        # Client initialisieren
        client = genai.Client(api_key=api_key)
        
        print(f"{'='*60}")
        print(f"🌐 Diagnose für API Key: {api_key[:4]}...{api_key[-4:]}")
        print(f"{'='*60}\n")

        # Modelle abrufen
        models = list(client.models.list())
        
        # Kategorien vorbereiten
        chat_models = []
        embed_models = []
        other_models = []

        for m in models:
            m_info = {
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description
            }
            
            if "embed" in m.name.lower():
                embed_models.append(m_info)
            elif "gemini" in m.name.lower():
                chat_models.append(m_info)
            else:
                other_models.append(m_info)

        # Ausgabe: Chat & Content Modelle
        print(f"🤖 CHAT & CONTENT MODELLE (für app.py):")
        for m in sorted(chat_models, key=lambda x: x['name']):
            print(f"  • {m['name']} ({m['display_name']})")
        
        print(f"\n🧬 EMBEDDING MODELLE (für Vektor-Suche):")
        if not embed_models:
            print("  ⚠️ Keine expliziten Embedding-Modelle gefunden!")
        for m in sorted(embed_models, key=lambda x: x['name']):
            print(f"  • {m['name']} ({m['display_name']})")

        print(f"\n🛠️ ANDERE MODELLE / EXPERIMENTELL:")
        for m in other_models:
            print(f"  • {m['name']}")

        print(f"\n{'='*60}")
        print("✅ Hinweis: Nutze in deinem Code immer den vollen Namen (inkl. 'models/').")

    except Exception as e:
        print(f"❌ Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    list_gemini_models()
