import streamlit as st
import os
import json
import pandas as pd
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sqlalchemy import create_engine
import warnings

# --- INITIALISIERUNG ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_search_results' not in st.session_state:
    st.session_state.last_search_results = []
if 'active_obj' not in st.session_state:
    st.session_state.active_obj = None
if 'quick_replies' not in st.session_state:
    st.session_state.quick_replies = []

# --- SETUP ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
PG_URI = "postgresql://donatusschmid@localhost:5432/vkm_new"
EMBEDDING_MODEL = "models/gemini-embedding-001"
AGENT_MODEL = "models/gemini-2.5-flash" 

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(layout="wide", page_title="VKM Agent Kurator", initial_sidebar_state="collapsed")

@st.cache_resource
def get_engine(): return create_engine(PG_URI)

@st.cache_resource
def get_client(): return genai.Client(api_key=API_KEY)

# --- HILFSFUNKTIONEN ---

def extract_quick_replies(text):
    """Extrahiert Sätze, die mit einem Fragezeichen enden, als Quick Replies."""
    # Findet Sätze am Ende des Textes, die ein ? enthalten
    questions = re.findall(r"([^.!?]+\?)", text)
    return [q.strip() for q in questions][-2:] # Die letzten zwei Fragen reichen meistens

def load_agent_prompt(specific_obj=None, vision_active=False):
    try:
        with open("system_prompt_agent.txt", "r", encoding="utf-8") as f:
            base = f.read()
    except:
        base = "Du bist der VKM AI-Kurator."
    
    if specific_obj:
        clean_obj = {k: str(v) for k, v in specific_obj.items() if k != 'embedding'}
        prompt = f"{base}\n\nAKTUELLER FOKUS AUF DIESES OBJEKT:\n{json.dumps(clean_obj, ensure_ascii=False)}"
        if vision_active:
            prompt += "\n\nVISUELLE ANALYSE AKTIV: Du erhältst das Bild des Objekts. Analysiere sichtbare Details, Symbole und Komposition."
        return prompt
    return base

# --- UI STYLING ---
st.markdown("""
    <style>
    .stChatMessage { background-color: #f8f9fb; border-radius: 12px; border: 1px solid #eef0f2; margin-bottom: 10px; }
    .gallery-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; background: white; margin-bottom: 15px; }
    .img-container { height: 120px; overflow: hidden; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; border-radius: 4px; margin-bottom: 8px; }
    .img-container img { max-height: 100%; max-width: 100%; object-fit: contain; }
    .fokus-banner { background: #34495e; color: white; padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 6px solid #e67e22; font-size: 0.9em; }
    .compact-detail-card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    .detail-img-box { width: 100%; height: 250px; overflow: hidden; border-radius: 4px; margin-bottom: 10px; background: #eee; display: flex; align-items: center; justify-content: center; }
    .detail-img-box img { width: 100%; height: 100%; object-fit: contain; }
    .desc-scroll { font-size: 0.9em; max-height: 180px; overflow-y: auto; color: #333; line-height: 1.4; }
    .quick-reply-btn { margin-right: 5px; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- DB TOOLS ---
def search_museum_database(query: str) -> str:
    client = get_client()
    engine = get_engine()
    try:
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
        ai_data = [{"inv": r['inventory_number'], "titel": r['title'], "info": str(r['description'])[:200]} for r in st.session_state.last_search_results]
        return json.dumps(ai_data, default=str)
    except Exception as e:
        return f"Fehler: {e}"

# --- LAYOUT ---
st.title("🏛️ VKM AI-Kurator")
col_chat, col_right = st.columns([1.6, 1.4])

with col_chat:
    if st.session_state.active_obj:
        st.markdown(f'<div class="fokus-banner"><strong>FOKUS:</strong> {st.session_state.active_obj["title"]}</div>', unsafe_allow_html=True)
        if st.button("⬅ Fokus aufheben", width="stretch"):
            st.session_state.active_obj = None
            st.session_state.quick_replies = []
            st.rerun()

    chat_box = st.container(height=580)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Quick Replies Anzeigen
    if st.session_state.quick_replies:
        cols = st.columns(len(st.session_state.quick_replies))
        for i, q in enumerate(st.session_state.quick_replies):
            if cols[i].button(f"💬 {q}", key=f"qr_{i}", width="stretch"):
                st.session_state.next_prompt = q
                st.rerun()

    # Chat Logik
    user_input = st.chat_input("Frage an den Kurator...")
    if hasattr(st.session_state, 'next_prompt'):
        user_input = st.session_state.next_prompt
        del st.session_state.next_prompt

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with chat_box:
            with st.chat_message("user"): st.markdown(user_input)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                
                # Checkbox Wert aus der rechten Spalte holen (via Session State)
                vision_active = st.session_state.get('vision_enabled', False)
                instruction = load_agent_prompt(st.session_state.active_obj, vision_active)
                
                content_parts = [types.Part(text=user_input)]
                if st.session_state.active_obj and vision_active:
                    img_url = st.session_state.active_obj.get('thumb_url', '').replace('width=150', 'width=800')
                    if img_url:
                        content_parts.append(types.Part.from_uri(file_uri=img_url, mime_type="image/jpeg"))

                try:
                    tools = [] if st.session_state.active_obj else [types.Tool(function_declarations=[
                        types.FunctionDeclaration(
                            name="search_museum_database",
                            description="Suche in der Datenbank",
                            parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                        )
                    ])]
                    
                    response = get_client().models.generate_content(
                        model=AGENT_MODEL, contents=content_parts, 
                        config=types.GenerateContentConfig(system_instruction=instruction, tools=tools)
                    )
                    
                    final_text = ""
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            with st.status("📡 Suche...") as s:
                                tool_res = search_museum_database(**part.function_call.args)
                                s.update(label="Daten geladen", state="complete")
                            final_resp = get_client().models.generate_content(model=AGENT_MODEL, contents=[types.Content(role="user", parts=content_parts), response.candidates[0].content, types.Content(role="tool", parts=[types.Part(function_response=types.FunctionResponse(name="search_museum_database", response={"result": tool_res}))])], config=types.GenerateContentConfig(system_instruction=instruction))
                            final_text = final_resp.text
                        elif part.text:
                            final_text = part.text

                    placeholder.markdown(final_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_text})
                    st.session_state.quick_replies = extract_quick_replies(final_text)
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler: {e}")

# --- RECHTE SPALTE ---
with col_right:
    if st.session_state.active_obj:
        obj = st.session_state.active_obj
        st.subheader("🔎 Objekt-Information")
        
        # Checkbox für Visuelle Analyse
        st.session_state.vision_enabled = st.checkbox("👁️ Visuelle Analyse (KI sieht das Bild)", value=False, help="Aktiviert die Bilderkennung für Detailfragen.")
        
        img_url = obj.get('thumb_url', '').replace('width=150', 'width=800') if obj.get('thumb_url') else "https://via.placeholder.com/400x300?text=Kein+Bild"
        st.markdown(f"""
            <div class="compact-detail-card">
                <div class="detail-img-box"><img src="{img_url}"></div>
                <div style="font-size:0.85em; color:#666; margin-bottom:5px;"><b>Inv. Nr.:</b> {obj.get('inventory_number', 'k.A.')}</div>
                <div style="font-weight: bold; margin-bottom: 5px;">{obj['title']}</div>
                <div class="desc-scroll">{obj.get('description', 'Keine Beschreibung vorhanden.')}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if obj.get('id'):
            st.link_button("🌐 Katalog-Link", f"https://www.volkskundemuseum.at/sammlungen/content/titleinfo/{obj['id']}", width="stretch")
    else:
        st.subheader("🖼️ Archiv-Fundstücke")
        res = st.session_state.last_search_results
        if res:
            for i in range(0, len(res), 3):
                cols = st.columns(3)
                for j in range(3):
                    if (idx := i + j) < len(res):
                        item = res[idx]
                        with cols[j]:
                            img = item['thumb_url'] if item['thumb_url'] else "https://via.placeholder.com/150"
                            st.markdown(f'<div class="gallery-card"><div class="img-container"><img src="{img}"></div><div style="font-size:0.7em; height:32px; overflow:hidden;">{item["title"][:40]}</div></div>', unsafe_allow_html=True)
                            if st.button("Fokus", key=f"f_{item['id']}", width="stretch"):
                                st.session_state.active_obj = item
                                st.session_state.quick_replies = []
                                st.rerun()
