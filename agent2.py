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
AGENT_MODEL = "gemini-2.5-flash" 

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="VKM Agent Kurator", initial_sidebar_state="collapsed")

# --- INITIALISIERUNG ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = []
if 'active_obj' not in st.session_state:
    st.session_state.active_obj = None

# --- DB & API CLIENT ---
@st.cache_resource
def get_engine(): return create_engine(PG_URI)

@st.cache_resource
def get_client(): return genai.Client(api_key=API_KEY)

def load_agent_prompt(specific_obj=None):
    """Lädt Instruktionen und erzwingt die Strukturierung von Archiv und Kontext."""
    try:
        with open("system_prompt_agent.txt", "r", encoding="utf-8") as f:
            base = f.read()
    except:
        base = "Du bist der VKM AI-Kurator. Trenne strikt in [Archiv] und Historischer Kontext."
    
    if specific_obj:
        # Konvertiere alle Metadaten (inkl. Timestamps) in Strings für JSON
        clean_obj = {k: str(v) for k, v in specific_obj.items()}
        return f"{base}\n\nFOKUS-OBJEKT: {json.dumps(clean_obj, ensure_ascii=False)}"
    
    return f"{base}\nNutze das Tool 'search_museum_database' für Archiv-Anfragen."

# --- CSS FÜR DAS LAYOUT ---
st.markdown("""
    <style>
    .stApp { height: 100vh; overflow: hidden; }
    
    /* Dialog Spalte (Links) */
    [data-testid="column"]:nth-child(1) {
        height: 80vh;
        display: flex;
        flex-direction: column;
        border-right: 1px solid #eee;
        padding-right: 20px;
    }
    
    /* Bilder Spalte (Rechts) */
    [data-testid="column"]:nth-child(2) {
        height: 80vh;
        overflow-y: auto;
        padding-left: 20px;
    }

    .header-hr { margin-top: 0; margin-bottom: 20px; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

# --- TOOLS ---
def search_museum_database(query: str) -> str:
    client = get_client()
    engine = get_engine()
    res = client.models.embed_content(model=EMBEDDING_MODEL, contents=query, config=types.EmbedContentConfig(output_dimensionality=768))
    vec = res.embeddings[0].values
    
    pattern = f"%{query}%"
    sql = """
    SELECT o.*, (1 - (e.embedding <=> %s::vector)) as similarity
    FROM objects o JOIN object_embeddings e ON o.id = e.object_id
    WHERE e.model = %s OR o.title ILIKE %s OR o.description ILIKE %s
    ORDER BY similarity DESC NULLS LAST LIMIT 9;
    """
    df = pd.read_sql_query(sql, engine, params=(vec, EMBEDDING_MODEL, pattern, pattern))
    st.session_state.last_search_results = df.to_dict('records')
    
    ai_data = [{"inv": r['inventory_number'], "titel": r['title'], "info": str(r['description'])[:150]} for r in st.session_state.last_search_results]
    return json.dumps(ai_data, default=str)

# --- HEADER (MIT LOGO) ---
head_col_logo, head_col_title = st.columns([1, 5])
with head_col_logo:
    logo_path = "vkm_logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=120)
    else:
        st.subheader("🏛️ VKM")

with head_col_title:
    st.markdown("<h2 style='margin-bottom: 0;'>Volkskundemuseum Wien Agent Kurator</h2>", unsafe_allow_html=True)

st.markdown("<div class='header-hr'></div>", unsafe_allow_html=True)

# --- UI LAYOUT ---
col_chat, col_gallery = st.columns([2, 1])

# --- LINKS: DIALOG ---
with col_chat:
    if st.session_state.active_obj:
        c1, c2 = st.columns([4, 1])
        with c1: st.markdown(f"**📍 Fokus:** {st.session_state.active_obj['title']}")
        with c2: 
            if st.button("Zurück ↩️", width="stretch"):
                st.session_state.active_obj = None
                st.rerun()
    else:
        st.markdown("**🌐 Modus:** Allgemeiner Archiv-Agent")

    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Fragen Sie den Kurator..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                placeholder = st.empty()
                client = get_client()
                
                try:
                    instruction = load_agent_prompt(st.session_state.active_obj)
                    tools = [] if st.session_state.active_obj else [types.Tool(function_declarations=[
                        types.FunctionDeclaration(
                            name="search_museum_database",
                            description="Sucht in der Museumsdatenbank nach Objekten und Themen.",
                            parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                        )
                    ])]

                    # API Aufruf
                    response = client.models.generate_content(
                        model=AGENT_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=instruction, 
                            tools=tools, 
                            temperature=0.85
                        )
                    )
                    
                    final_text = ""
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            fn_args = part.function_call.args
                            placeholder.markdown(f"📡 *Suche im Archiv nach '{fn_args['query']}'...*")
                            
                            tool_res = search_museum_database(**fn_args)
                            
                            # Finaler Text mit Tool-Ergebnissen
                            final_resp = client.models.generate_content(
                                model=AGENT_MODEL,
                                contents=[
                                    types.Content(role="user", parts=[types.Part(text=prompt)]),
                                    response.candidates[0].content,
                                    types.Content(role="tool", parts=[types.Part(
                                        function_response=types.FunctionResponse(name="search_museum_database", response={"result": tool_res})
                                    )])
                                ],
                                config=types.GenerateContentConfig(system_instruction=instruction, temperature=0.85)
                            )
                            final_text = final_resp.text
                            st.session_state.chat_history.append({"role": "assistant", "content": final_text})
                            st.rerun()
                        elif part.text:
                            final_text += part.text
                    
                    if final_text:
                        placeholder.markdown(final_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": final_text})

                except Exception as e:
                    st.error(f"Fehler: {e}")

# --- RECHTS: BILDER-GALERIE (3 pro Zeile) ---
with col_gallery:
    st.subheader("🖼️ Archiv-Treffer")
    results = st.session_state.last_search_results
    if results:
        # Grid-Layout: 3 Bilder pro Zeile
        for i in range(0, len(results), 3):
            cols = st.columns(3)
            for j in range(3):
                idx = i + j
                if idx < len(results):
                    item = results[idx]
                    with cols[j]:
                        st.image(item['thumb_url'] if item['thumb_url'] else "https://via.placeholder.com/150", width="stretch")
                        if st.button("🔎", key=f"btn_{item['id']}", width="stretch", help=f"Fokus auf: {item['title']}"):
                            st.session_state.active_obj = item
                            st.session_state.chat_history.append({"role": "assistant", "content": f"Experten-Dialog zu: **{item['title']}**"})
                            st.rerun()
    else:
        st.info("Keine Treffer vorhanden.")
