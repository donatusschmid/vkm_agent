import streamlit as st
import os
import json
import pandas as pd
import re
import time
from google import genai
from google.genai import types
from google.genai.errors import ClientError
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

def load_agent_prompt(specific_obj=None):
    try:
        with open("system_prompt_agent.txt", "r", encoding="utf-8") as f:
            base = f.read()
    except:
        base = "Du bist der VKM AI-Kurator. Trenne strikt in [Archiv] und Historischer Kontext."
    
    if specific_obj:
        clean_obj = {k: str(v) for k, v in specific_obj.items() if k != 'embedding'}
        return f"{base}\n\nAKTUELLER FOKUS AUF DIESES OBJEKT:\n{json.dumps(clean_obj, ensure_ascii=False)}"
    return base

# --- UI STYLING (Standard Chat & Galerie) ---
st.markdown("""
    <style>
    .stChatMessage { background-color: #f8f9fb; border-radius: 12px; border: 1px solid #eef0f2; margin-bottom: 10px; }
    .gallery-card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; background: white; margin-bottom: 15px; }
    .img-container { height: 120px; overflow: hidden; display: flex; align-items: center; justify-content: center; background-color: #f0f0f0; border-radius: 4px; margin-bottom: 8px; }
    .img-container img { height: 100%; width: 100%; object-fit: cover; }
    .fokus-banner { background: #34495e; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 8px solid #e67e22; }
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

col_chat, col_gallery = st.columns([1.8, 1.2])

with col_chat:
    if st.session_state.active_obj:
        obj = st.session_state.active_obj
        st.markdown(f'<div class="fokus-banner"><small>FOKUS</small><br><strong>{obj["title"]}</strong><br><code>{obj["inventory_number"]}</code></div>', unsafe_allow_html=True)
        if st.button("Fokus aufheben"):
            st.session_state.active_obj = None
            st.rerun()

    chat_box = st.container(height=600)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Frage an den Kurator..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                instruction = load_agent_prompt(st.session_state.active_obj)
                
                try:
                    tools = [] if st.session_state.active_obj else [types.Tool(function_declarations=[
                        types.FunctionDeclaration(
                            name="search_museum_database",
                            description="Suche in der Datenbank",
                            parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]}
                        )
                    ])]

                    response = get_client().models.generate_content(
                        model=AGENT_MODEL,
                        contents=prompt,
                        config=types.GenerateContentConfig(system_instruction=instruction, tools=tools)
                    )
                    
                    final_text = ""
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            with st.status("📡 Suche...") as s:
                                tool_res = search_museum_database(**part.function_call.args)
                                s.update(label="Daten geladen", state="complete")
                            
                            final_resp = get_client().models.generate_content(
                                model=AGENT_MODEL,
                                contents=[
                                    types.Content(role="user", parts=[types.Part(text=prompt)]),
                                    response.candidates[0].content,
                                    types.Content(role="tool", parts=[types.Part(function_response=types.FunctionResponse(name="search_museum_database", response={"result": tool_res}))])
                                ],
                                config=types.GenerateContentConfig(system_instruction=instruction)
                            )
                            final_text = final_resp.text
                        elif part.text:
                            final_text = part.text

                    placeholder.markdown(final_text)
                    st.session_state.chat_history.append({"role": "assistant", "content": final_text})
                    st.rerun()
                except Exception as e:
                    st.error(f"Fehler: {e}")

with col_gallery:
    st.subheader("🖼️ Fundstücke")
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
                        if st.button("Fokus", key=f"f_{item['id']}", use_container_width=True):
                            st.session_state.active_obj = item
                            st.rerun()
