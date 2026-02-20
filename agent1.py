import streamlit as st
import os
import json
import pandas as pd
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sqlalchemy import create_engine
import warnings

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
PG_URI = "postgresql://donatusschmid@localhost:5432/vkm_new"
EMBEDDING_MODEL = "models/gemini-embedding-001"
AGENT_MODEL = "gemini-2.0-flash" 

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="VKM Agentic Explorer")

# --- DB & API CLIENT ---
@st.cache_resource
def get_engine():
    return create_engine(PG_URI)

@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)

def load_agent_prompt():
    """Lädt die erweiterten Instruktionen für hybrides Wissen."""
    try:
        with open("system_prompt_agent.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return """Du bist der VKM AI-Kurator. 
        TRENNUNG DER QUELLEN:
        - Markiere Fakten aus der Datenbank mit [Archiv].
        - Markiere dein eigenes Wissen mit 'Historischer Kontext:'.
        - Wenn die Datenbank keine Treffer liefert, suche nach verwandten Themen.
        - Beende JEDE Antwort mit einem Vorschlag für eine weiterführende Frage an den Nutzer."""

# --- ERWEITERTES TOOL: Suche mit Fallback ---
def search_museum_database(query: str) -> str:
    """Sucht in der Museumsdatenbank nach Objekten, Künstlern und verwandten Themen."""
    client = genai.Client(api_key=API_KEY)
    engine = create_engine(PG_URI)
    
    try:
        # Vektorisierung der Anfrage
        res = client.models.embed_content(model=EMBEDDING_MODEL, contents=query)
        vec = res.embeddings[0].values
        
        search_pattern = f"%{query}%"
        sql = """
        SELECT o.id, o.title, o.description, o.inventory_number, o.thumb_url,
               (1 - (e.embedding <=> %s::vector)) as similarity
        FROM objects o
        JOIN object_embeddings e ON o.id = e.object_id
        WHERE e.model = %s 
           OR o.title ILIKE %s 
           OR o.description ILIKE %s
        ORDER BY similarity DESC NULLS LAST
        LIMIT 6;
        """
        df = pd.read_sql_query(sql, engine, params=(vec, EMBEDDING_MODEL, search_pattern, search_pattern))
        
        # Fallback: Wenn nichts gefunden wurde, versuchen wir eine breitere Suche
        if df.empty or (len(df) < 2):
            # Die KI liefert oft sehr spezifische Queries, wir kürzen hier auf das erste Wort
            broad_query = query.split()[0] if len(query.split()) > 0 else query
            df = pd.read_sql_query(sql, engine, params=(vec, EMBEDDING_MODEL, f"%{broad_query}%", f"%{broad_query}%"))

        if df.empty:
            return "Keine Treffer im Archiv gefunden."
        
        st.session_state.last_search_results = df.to_dict('records')
        
        results_for_ai = []
        for _, row in df.iterrows():
            results_for_ai.append({
                "inv": row['inventory_number'],
                "titel": row['title'],
                "info": (row['description'][:150] + "..") if row['description'] else ""
            })
        return json.dumps(results_for_ai, ensure_ascii=False)
    except Exception as e:
        return f"Fehler: {str(e)}"

# --- INITIALISIERUNG ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = []

# --- UI ---
st.title("🏛️ VKM Agentic Explorer (Hybrid)")

col_chat, col_gallery = st.columns([2, 1])

with col_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Fragen Sie etwas zur Sammlung..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            client = get_client()
            available_functions = {"search_museum_database": search_museum_database}
            
            def call_gemini_with_retry(contents, config, max_retries=3):
                for i in range(max_retries):
                    try:
                        return client.models.generate_content(model=AGENT_MODEL, contents=contents, config=config)
                    except Exception as e:
                        if "429" in str(e) and i < max_retries - 1:
                            time.sleep((i + 1) * 5)
                        else:
                            raise e

            try:
                # 1. Konfiguration & Erstanfrage
                config = types.GenerateContentConfig(
                    system_instruction=load_agent_prompt(),
                    temperature=0.7, # Höhere Kreativität für Kontextwissen
                    tools=[types.Tool(function_declarations=[
                        types.FunctionDeclaration(
                            name="search_museum_database",
                            description="Sucht im Museumsarchiv nach Objekten, Personen und Themenkontexten.",
                            parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                        )
                    ])]
                )

                response = call_gemini_with_retry(prompt, config)

                # 2. Part-Handling & Tool-Integration
                final_text = ""
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        fn_name = part.function_call.name
                        fn_args = part.function_call.args
                        placeholder.markdown(f"📡 *Archiv-Recherche läuft: {fn_args['query']}...*")
                        
                        tool_result = available_functions[fn_name](**fn_args)
                        
                        # Ergebnis zurück an KI
                        final_response = call_gemini_with_retry(
                            contents=[
                                types.Content(role="user", parts=[types.Part(text=prompt)]),
                                response.candidates[0].content,
                                types.Content(role="tool", parts=[
                                    types.Part(
                                        function_response=types.FunctionResponse(
                                            name=fn_name,
                                            response={"result": tool_result}
                                        )
                                    )
                                ])
                            ],
                            config=types.GenerateContentConfig(system_instruction=load_agent_prompt(), temperature=0.7)
                        )
                        final_text = final_response.text
                    elif part.text:
                        final_text += part.text

                placeholder.markdown(final_text if final_text else "Keine Antwort möglich.")
                st.session_state.chat_history.append({"role": "assistant", "content": final_text})
                
                if st.session_state.last_search_results:
                    st.rerun()

            except Exception as e:
                placeholder.error(f"Fehler: {e}")

# --- GALERIE ---
with col_gallery:
    st.subheader("🖼️ Datenbank-Treffer")
    if st.session_state.last_search_results:
        for item in st.session_state.last_search_results:
            with st.container(border=True):
                st.image(item['thumb_url'] if item['thumb_url'] else "https://via.placeholder.com/150")
                st.markdown(f"**{item['title']}**")
                st.caption(f"Inv: {item['inventory_number']}")
    else:
        st.info("Hier erscheinen relevante Objekte aus der Datenbank.")
