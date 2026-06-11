import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd
import time
import os
import sys
import plotly.express as px
import requests
import tempfile
import re
import fitz
import logging
import glob
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from backend.rag.image_store import metadata
from backend.rag.memory_store import get_all_memory, retrieve_memory, get_memory_count, embedder

os.environ["HF_TOKEN"] = ""
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
FRONTEND_DIR = os.path.dirname(__file__)

USER_DB = os.path.join(
    FRONTEND_DIR,
    "users.csv"
)
def validate_password(password: str) -> tuple[bool, str]:
    if not password:
        return False, "Password cannot be empty."
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)."
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number (0-9)."
    if not re.search(r"[@$!%*?&#^_-]", password):
        return False, "Password must contain at least one special character."
    return True, "Password is valid."

if not os.path.exists(USER_DB):
    df = pd.DataFrame(columns=["username", "password", "role", "reg_date"])
    # Add a default admin so you don't get locked out
    admin_data = pd.DataFrame([{"username": "admin", "password": "admin123", "role": "admin", "reg_date": "2026-01-01"}])
    df = pd.concat([df, admin_data], ignore_index=True)
    df.to_csv(USER_DB, index=False)

def register_user(username, password):
    is_valid, msg = validate_password(password)
    if not is_valid:
        return False, f"Registration failed: {msg}"
    df = pd.read_csv(USER_DB)
    if username in df['username'].values:
        return False, "Username already exists!"
    
    new_user = pd.DataFrame([{
        "username": username, 
        "password": password, 
        "role": "user", 
        "reg_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    }])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB, index=False)
    return True, "Registration successful!"

if "theme" not in st.session_state:
    st.session_state.theme = "light"  

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

if "last_agent" not in st.session_state:
    st.session_state.last_agent = "unknown"

if "result" not in st.session_state:
    st.session_state.result = None

if "show_repetition_warning" not in st.session_state:
    st.session_state.show_repetition_warning = False


st.set_page_config(
    page_title="Multimodal AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"  # Changed from "dark" to "light"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if st.session_state.theme == "dark":
    primary_grad = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    card_bg = "rgba(30, 41, 59, 0.7)"
    text_col = "#f8fafc"
    border_col = "rgba(255, 255, 255, 0.1)"
    header_bg = "rgba(15, 23, 42, 0.8)"
else:
    primary_grad = "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)"
    card_bg = "rgba(255, 255, 255, 0.7)"
    text_col = "#0f172a"
    border_col = "rgba(0, 0, 0, 0.05)"
    header_bg = "rgba(255, 255, 255, 0.8)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    .stApp {{
        background: {primary_grad};
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    /* Glassmorphism Card Effect */
    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        border: 1px solid {border_col};
        border-radius: 24px;
        padding: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }}

    /* Header Styling */
    .nav-bar {{
        background: {header_bg};
        backdrop-filter: blur(10px);
        padding: 10px 40px;
        border-bottom: 1px solid {border_col};
        position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
        display: flex; justify-content: space-between; align-items: center;
    }}

    .main-title {{
        font-size: 52px; font-weight: 800; text-align: center;
        background: linear-gradient(90deg, #3b82f6, #2dd4bf);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }}

    /* Input focus effects */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background: transparent !important;
        border-radius: 12px !important;
        border: 1px solid {border_col} !important;
        color: {text_col} !important;
    }}

    /* Button hover animations */
    .stButton>button {{
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
    }}
</style>
""", unsafe_allow_html=True)


def login_page():
    # Subtle background decorative elements
    st.markdown('<div style="position:fixed; top:-100px; left:-100px; width:400px; height:400px; background:rgba(59,130,246,0.1); filter:blur(100px); border-radius:50%;"></div>', unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 1.5, 1])
    with center_col:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
        st.markdown('<div class="main-title">Multimodal AI Agent System</div>', unsafe_allow_html=True)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        
        mode = st.radio("Access Level", ["🔐 Login", "📝 Register"], horizontal=True, label_visibility="collapsed")
        
        if mode == "🔐 Login":
            st.markdown(f"<h2 style='color:{text_col}; margin-bottom:20px;'>Welcome Back</h2>", unsafe_allow_html=True)
            u = st.text_input("Username", placeholder="e.g. admin")
            p = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True, type="primary"):
                df = pd.read_csv(USER_DB)
                user_match = df[(df['username'] == u) & (df['password'] == p)]
                if not user_match.empty:
                    st.session_state.logged_in = True
                    st.session_state.user_role = user_match.iloc[0]['role']
                    st.session_state.username = u
                    st.rerun()
                else: st.error("Access Denied: Invalid Credentials")
        else:
            st.markdown(f"<h2 style='color:{text_col}; margin-bottom:20px;'>Create Account</h2>", unsafe_allow_html=True)
            new_u = st.text_input("New Username")
            new_p = st.text_input("Password", type="password")
            conf_p = st.text_input("Confirm Password", type="password")
            password_verified = False
            if new_p:
                is_valid, msg = validate_password(new_p)
                if not is_valid:
                    st.error(f"⚠️ {msg}")
                elif new_p != conf_p:
                    st.warning("⚠️ Passwords do not match.")
                else:
                    st.success("🔒 Password Matched!")
                    password_verified = True

            if st.button("Complete Registration", use_container_width=True):
                # Guard checks to ensure clean form submission
                if not new_u.strip():
                    st.error("⚠️ Username cannot be empty!")
                elif not new_p:
                    st.error("⚠️ Password cannot be empty!")
                elif new_p != conf_p:
                    st.error("⚠️ Passwords do not match.")
                elif not password_verified:
                    st.error("⚠️ Please choose a stronger password matching the rules above.")
                else:
                    success, msg = register_user(new_u, new_p)
                    if success: st.success(msg)
                    else: st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.logged_in:
    login_page()
else:


    header_col1, header_col2, header_col3 = st.columns([2, 5, 2])
    with header_col1:
    # Move Logo to Header
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; padding-top: 5px;">
            <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" width="30">
            <span style="font-weight: 700; font-size: 18px;">Multimodal.ai</span>
            </div>
        """, unsafe_allow_html=True)

    with header_col2:
    # Navigation Buttons 
        if st.session_state.user_role == "admin":
            tabs = ["Dashboard", "Analytics", "View Logs", "Feedback", "Users"]
        else:
            tabs = ["Dashboard", "Feedback", "View Logs", "About us"]
       
        nav_cols = st.columns(len(tabs))
    
        for i, tab_name in enumerate(tabs):

            icons = {
            "Dashboard": "🏠", "Analytics": "📊", "View Logs": "🧪" 
             ,"Users": "👥", "Feedback": "⭐", "About us": "🏢"
            }
            button_label = f"{icons.get(tab_name, '')} {tab_name}"
        
            if nav_cols[i].button(button_label, use_container_width=True):
                st.session_state.active_tab = tab_name

    with header_col3:

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if st.button("🌓"):
                st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
                st.rerun()
        with t_col2:
            st.write(f"👤 Welcome **{st.session_state.username}**")
            if st.button("Logout", key="my_unique_button"):
                st.session_state.logged_in = False
                st.rerun()
    st.markdown("""
    <style>
    .st-key-my_unique_button button {
        padding-top:10px;
        font-weight:600;
        color:#ef4444;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("---")


# MAIN CONTENT AREA


    _, center_col, _ = st.columns([1, 8, 1])

    with center_col:
        if st.session_state.active_tab == "Dashboard":
            st.markdown('<div class="main-title">Multimodal AI Agent System</div>', unsafe_allow_html=True)
            st.markdown('<p style="text-align:center; color:#64748b; margin-bottom:40px;">Post-Trained LLM • Multi-Agent Architecture • Enterprise Ready</p>', unsafe_allow_html=True)
        
       

            body_left, body_right = st.columns([1.5, 1], gap="large")

            with body_left:
                st.write("### AI Agent System")
                user_input = st.text_area("Type your request", height=150, placeholder="Describe the analysis you need...")
                run = st.button("Run AI Agent", type="primary", use_container_width=True)

            with body_right:
                uploaded_file = st.file_uploader(
                    "Upload PDF / Image",
                    type=["pdf", "jpg", "jpeg", "png"]
                )
                camera_image = st.camera_input("Live Camera")

            if run:

                files = {}
                data = {}

                if user_input:
                    data["text"] = user_input

                st.session_state.current_lang = "en"

                if uploaded_file is not None:
                    files["file"] = (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                    pdf_doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                    raw_extracted_text = "".join([page.get_text() for page in pdf_doc])
                else:
                    raw_extracted_text = user_input
                st.session_state.original_text = raw_extracted_text
                if camera_image is not None:
                    files["image"] = (
                        "camera.jpg",
                        camera_image.getvalue(),
                        "image/jpeg"
                    )

                response = requests.post(
                    "http://127.0.0.1:8000/route",
                    data=data,
                    files=files
                )

                if response.status_code == 200:
                    try:
                        result = response.json()
                        st.session_state.result = result
                    except Exception as e:
                        st.error(f"Invalid JSON response: {response.text}")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")


            result = st.session_state.result

            if result:

                st.success(f"🧠 Routed to: {result.get('agent')}")
                
                if result.get("agent") == "vision":
                    st.subheader("🔍 Detected Objects")

                    if "image" in result:
                        st.image(base64.b64decode(result["image"]))

                    st.markdown("#### 📋 Objects Found")
                    for obj in result.get("objects", []):
                        st.markdown(f"**{obj['label']}** • `{obj['confidence']}`")


                elif result.get("agent") == "tts":
                    st.subheader("🔊 Audio Output")
                    LANGUAGES = {
                        "English": "en",
                        "Hindi (हिन्दी)": "hi",
                        "Bengali (বাংলা)": "bn",
                        "Telugu (తెలుగు)": "te",
                        "Marathi (मराठी)": "mr",
                        "Tamil (தமிழ்)": "ta",
                        "Gujarati (ગુજરાતી)": "gu",
                        "Kannada (ಕನ್ನಡ)": "kn",
                        "Malayalam (മലയാളം)": "ml",
                        "Urdu (اردو)": "ur",
                        "Spanish (Español)": "es",
                        "French (Français)": "fr",
                        "German (Deutsch)": "de",
                        "Japanese (日本語)": "ja"
                    }
                    if "current_lang" not in st.session_state:
                        st.session_state.current_lang = "en"
                    current_lang = st.session_state.get("current_lang", "en")
                    lang_values = list(LANGUAGES.values())
        
                    if current_lang in lang_values:
                        default_index = lang_values.index(current_lang)
                    else:
                        default_index = 0
                    selected_lang_name = st.selectbox(
                        "Select Audio Language", 
                        options=list(LANGUAGES.keys()), 
                        index=list(LANGUAGES.values()).index(st.session_state.current_lang)
                    )
                    selected_lang_code = LANGUAGES[selected_lang_name]
                    if selected_lang_code != st.session_state.current_lang:
                        st.session_state.current_lang = selected_lang_code
                        
                        
                        try:
                           
                            updated_payload = {
                                "text": user_input,
                                "lang": selected_lang_code
                            }
                            
                            
                            response = requests.post("http://127.0.0.1:8000/route", data=updated_payload)
                            if response.status_code == 200:
                                st.session_state.result = response.json()
                                st.rerun()  # Instantly refresh the page to play the new audio
                            else:
                                st.error("Failed to update language on backend.")
                        except Exception as e:
                            st.error(f"Error fetching new language: {str(e)}")
                    audio_b64 = result.get("audio")

                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        st.audio(audio_bytes, format="audio/wav")
                    else:
                        st.error("No audio received")

                elif result.get("agent") == "summarizer":
                    if "output" in result and isinstance(result["output"], dict):
                        payload = result["output"]
                    elif "data" in result and isinstance(result["data"], dict):
                        payload = result["data"]
                    else:
                        payload = result
                    st.subheader("📄 Summary")
                    summary_text = payload.get("summary", payload.get("output", payload.get("text", "No summary text found in response.")))
                    st.write(summary_text)
                    st.markdown("---")

                    col1, col2, col3 = st.columns(3)

                    with col1:
  
                        conf = payload.get("confidence", 0.0)
                        st.metric("Confidence Score", f"{float(conf) * 100:.1f}%")

                    with col2:
   
                        if payload.get("hallucinated", False):
                            h_ratio = payload.get("hallucination_ratio", 0.0)
                            st.error(f"Hallucination Risk ⚠️ ({float(h_ratio) * 100:.0f}%)")
                        else:
                            st.success("Low Hallucination Risk ✅")

                    with col3:
  
                        f1_score = payload.get("bertscore_f1", 0.0)
                        st.metric("Accuracy", f"{float(f1_score) * 100:.1f}%")


                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.write(f"Precision: `{payload.get('bertscore_precision', 0.0)*100:.1f}`")

                    with c2:

                        st.write(f"Recall: `{payload.get('bertscore_recall', 0.0)*100:.1f}`")

                    with c3:

                        st.write(f"F1 Score: `{payload.get('bertscore_f1', 0.0)*100:.1f}`")

                    st.markdown("---")
                    st.subheader("💬 Interactive Document Q&A")
                    st.caption("Ask questions about the document content uploaded, the RAG agent will give the answer.")

             
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []

              
                    for message in st.session_state.chat_history:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            if "sources" in message and message["sources"]:

                                st.info(message["sources"])

        
                    question = st.chat_input("What did the document say about...")
                    if question :
        
                        original_doc = st.session_state.get("original_text", "")
                        
                        if not original_doc:
                            st.error("Please run a summary on a document first before asking questions!")
                        else:
   
                            with st.chat_message("user"):
                                st.markdown(question)
                            st.session_state.chat_history.append({"role": "user", "content": question})


                            with st.spinner("RAG Agent searching document..."):
                                try:
                                    import requests
                                    # Replace withactual FastAPI URL if deployed
                                    BACKEND_URL = "http://localhost:8000/qa" 
                                    
                                    response = requests.post(
                                        BACKEND_URL, 
                                        json={"question": question, "document_text": original_doc}
                                    )
                                    
                                    if response.status_code == 200:
                                        res_data = response.json()
                                        answer = res_data.get("answer", "No response generated.")
                                        sources = res_data.get("sources", "")

                                        # Display system response
                                        with st.chat_message("assistant"):
                                            st.markdown(answer)
                                            if sources:
                                                
                                                st.info(sources)

                                        # Save response to history
                                        st.session_state.chat_history.append({
                                            "role": "assistant",
                                            "content": answer,
                                            "sources": sources
                                        })
                                    else:
                                        st.error("Error communicating with Q&A backend.")
                                except Exception as e:
                                    st.error(f"Connection failed: {e}")
            if run and not user_input and uploaded_file is None and camera_image is None:
                st.warning("Please provide input")
                st.stop()

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("### 🚀 Core Platform Pillars")
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.markdown(f"""
                <div style="background:{card_bg}; padding:20px; border-radius:12px; border:1px solid {border_col}; margin-bottom:20px;">
                    <h4 style="margin-top:0;">🧠 Post-Trained Intelligence</h4>
                    <p style="font-size:18px; color:#64748b;">Our models are specifically post-trained on domain-specific datasets to ensure higher reasoning capabilities and lower hallucination rates compared to base models.</p>
                </div>
                <div style="background:{card_bg}; padding:20px; border-radius:12px; border:1px solid {border_col};">
                    <h4 style="margin-top:0;">🔗 Multimodal Fusion</h4>
                    <p style="font-size:18px; color:#64748b;">Seamlessly process and correlate data across text, images, and documents. Our agents don't just see data; they understand the context between different formats.</p>
                </div>
                """, unsafe_allow_html=True)
        
            with f_col2:
                st.markdown(f"""
                <div style="background:{card_bg}; padding:20px; border-radius:12px; border:1px solid {border_col}; margin-bottom:20px;">
                    <h4 style="margin-top:0;">🤖 Autonomous Agent Orchestration</h4>
                    <p style="font-size:18px; color:#64748b;">A sophisticated multi-agent architecture where specialized "sub-agents" collaborate to solve complex, multi-step enterprise tasks.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)
 

            st.markdown("### 🔄 How It Works")
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                st.markdown("**1. Ingest**")
                st.markdown("Upload your documents, images or raw text prompts into the unified interface.")
            with w_col2:
                st.markdown("**2. Route**")
                st.markdown("Our Orchestrator Agent analyzes the request and routes it to the most qualified specialized model.")
            with w_col3:
                st.markdown("**3. Refine**")
                st.markdown("The system cross-references the output across modalities to ensure accuracy and provides a structured, actionable summary.")

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")

        elif st.session_state.active_tab == "Analytics" and st.session_state.user_role == "admin":
            st.markdown("### 📊 Evaluation Dashboard")
    
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Current Model Version", "v2 LoRA")
            kpi2.metric("Routing Accuracy", "91.4%", "+2.1%")
            kpi3.metric("Avg User Rating", "4.3 / 5.0", "+0.4")
    
            st.divider()

            METRICS_FILE = os.path.join(os.path.dirname(__file__), "../backend/evaluation/metrics.csv")

            if os.path.exists(METRICS_FILE):
                df = pd.read_csv(METRICS_FILE)
                if "hallucination" in df.columns:
                    df["hallucination"] = (
                        df["hallucination"]
                        .astype(str)
                        .str.lower()
                        .map({
                            "true": True,
                            "false": False,
                            "1": True,
                            "0": False
                        })
                    )

                st.dataframe(df)
                if len(df) > 0:
                    st.markdown("#### 📈 System Analytics")
            

                    chart_col1, chart_col2 = st.columns(2)
            

                    with chart_col1:
                        fig_conf = px.line(df, y="confidence", title="Confidence Over Time", markers=True, 
                                   color_discrete_sequence=["#3b82f6"])
                        fig_conf.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_conf, width="stretch")
            
  
                    with chart_col2:
                        fig_lat = px.line(df, y="latency", title="Latency Over Time (sec)", markers=True,
                                  color_discrete_sequence=["#ef4444"])
                        fig_lat.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_lat, width="stretch")

  
                    st.markdown("<br>", unsafe_allow_html=True)
                    agent_counts = df["agent"].value_counts().reset_index()
                    agent_counts.columns = ["Agent", "Count"]
                    fig_bar = px.bar(agent_counts, x="Agent", y="Count", title="Agent Usage Breakdown",
                             color="Agent", text="Count")
                    fig_bar.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
                    st.plotly_chart(fig_bar, width="stretch")

                    with st.expander("View Raw Activity Log", expanded=False):
                        st.dataframe(df.tail(10), width="stretch")

                    if "model_version" in df.columns:
                        st.divider()
                        st.markdown("#### 🧠 Model Version Analytics")
                        v_col1, v_col2 = st.columns(2)
                        with v_col1:
                            version_conf = df.groupby("model_version")["confidence"].mean().reset_index()
                            fig_v_conf = px.bar(version_conf, x="model_version", y="confidence", title="Avg Confidence by Model")
                            st.plotly_chart(fig_v_conf, width="stretch")

                        with v_col2:
                            version_latency = df.groupby("model_version")["latency"].mean().reset_index()
                            fig_v_lat = px.bar(version_latency, x="model_version", y="latency", title="Avg Latency by Model")
                            st.plotly_chart(fig_v_lat, width="stretch")
                else:
                    st.info("Metrics file exists but contains no data.")
            else:
                st.markdown("<div class='skeleton-box'></div>", unsafe_allow_html=True)
                st.caption("Waiting for telemetry data...")

            st.divider()

        elif st.session_state.active_tab == "View Logs":
            st.markdown("### 🧪 View Memory Logs")
            st.header("🧠 Vector Memory Database")

            memory_count = get_memory_count()

            st.metric(
                "Stored Memory Vectors",
                memory_count
            )


            st.subheader("📚 Stored Memories")

            all_memory = get_all_memory()

            if len(all_memory) == 0:
                st.info("No memory stored yet.")

            else:

                for i, mem in enumerate(all_memory[-10:]):

                    st.markdown(f"""
                    ### Memory {i+1}

                    {mem}

                    ---
                    """)

            st.subheader("🔎 Semantic Search")

            query = st.text_input(
                "Search Vector Memory"
            )

            if query:

                results = retrieve_memory(query)

                st.subheader("🎯 Retrieved Context")

                if len(results) == 0:
                    st.warning("No relevant memory found")

                else:

                    for r in results:

                        st.markdown(f"""
                        ### Similar Memory

                        **Text:**
                        {r['text']}

                        **Similarity Distance:**
                        {round(r['distance'], 3)}

                        ---
                        """)
            st.markdown("---")
            st.subheader("🚨  Detection Logs")

            IMAGE_DIR = os.path.join(
                ROOT_DIR,
                "backend",
                "storage",
                "person_detections"
            )

            os.makedirs(
                IMAGE_DIR,
                exist_ok=True
            )


            image_paths = glob.glob(
                os.path.join(IMAGE_DIR, "*.jpg")
            )


            st.metric(
                "Stored Detection Images",
                len(image_paths)
            )


            if len(image_paths) == 0:

                st.warning(
                    "No person detections stored yet."
                )


            else:

                st.success(
                    f"{len(image_paths)} person detections found."
                )

                cols = st.columns(3)

                for i, img_path in enumerate(
                    reversed(image_paths)
                ):

                    with cols[i % 3]:

                        st.image(
                            img_path,
                            use_container_width=True
                        )

                        st.caption(
                            os.path.basename(img_path)
                        )

                        # DOWNLOAD BUTTON

                        with open(img_path, "rb") as file:

                            st.download_button(

                                label="Download",

                                data=file,

                                file_name=os.path.basename(
                                    img_path
                                ),

                                mime="image/jpeg",

                                key=img_path
                            )

        
        elif st.session_state.active_tab == "Feedback" and st.session_state.user_role == "user":
            st.markdown("### ⭐ Rate System Output")
            with st.container():
                feedback_text = st.text_area("📝 Additional Comments (Optional)", placeholder="Tell us what you liked or what should be improved...")
                rating = st.slider("Quality Rating", 1, 5, 5)

                if st.button("Submit Feedback"):
                    feedback_data = {
                        "input_text": st.session_state.last_input,
                        "rating": rating,
                        "comment": feedback_text,
                        "agent": st.session_state.last_agent
                    }

                    r = requests.post("http://127.0.0.1:8000/feedback", json=feedback_data)
                    if r.status_code == 200:
                        st.toast('Feedback submitted successfully! Thank you.', icon='🚀')
                    else:
                        st.toast('Failed to submit feedback.', icon='❌')

        elif st.session_state.active_tab == "Feedback" and st.session_state.user_role == "admin":
            def load_feedback_data():
                FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "../backend/evaluation/feedback.csv")
                if os.path.exists(FEEDBACK_FILE):
                    try:
                        df = pd.read_csv(FEEDBACK_FILE, names=["timestamp", "input_text", "rating", "comment"], skiprows=1)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        return df
                    except Exception as e:
                        st.error(f"Error reading feedback: {e}")
                        return pd.DataFrame()
                return pd.DataFrame()

            # --- ADMIN FEEDBACK ANALYSIS ---
            st.markdown("### 📊 Feedback Overview")
            feedback_df = load_feedback_data()

            if not feedback_df.empty:
   
                total_feedbacks = len(feedback_df)
                avg_rating = feedback_df['rating'].mean()
                positive_count = len(feedback_df[feedback_df['rating'] >= 4])
                negative_count = total_feedbacks - positive_count
                pos_pct = (positive_count / total_feedbacks) * 100

    
                kpi1, kpi2, kpi3 = st.columns(3)
        
                with kpi1:
                    st.markdown(f"""
                        <div class="glass-card" style="padding:20px; text-align:center;">
                            <p style="color:#64748b; margin:0; font-weight:600;">Average Rating</p>
                            <h2 style="margin:0; color:#3b82f6;">{avg_rating:.2f} / 5.0</h2>
                        </div>
                    """, unsafe_allow_html=True)
        
                with kpi2:
                    st.markdown(f"""
                        <div class="glass-card" style="padding:20px; text-align:center;">
                        <p style="color:#64748b; margin:0; font-weight:600;">Total Feedbacks Received</p>
                        <h2 style="margin:0; color:#10b981;">{total_feedbacks}</h2>
                    </div>
                """, unsafe_allow_html=True)

                with kpi3:
                    st.markdown(f"""
                        <div class="glass-card" style="padding:20px; text-align:center;">
                            <p style="color:#64748b; margin:0; font-weight:600;">Positive Sentiment</p>
                            <h2 style="margin:0; color:#f59e0b;">{pos_pct:.1f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

        
        # Prepare data 
                sentiment_data = pd.DataFrame({
                    "Category": ["Positive (4-5 ⭐)", "Negative (1-3 ⭐)"],
                    "Count": [positive_count, negative_count],
                    "Color": ["#10b981", "#ef4444"] # Green for positive, Red for negative
                })

                fig = px.bar(
                    sentiment_data, 
                    x="Category", 
                    y="Count",
                    text="Count",
                    color="Category",
                    color_discrete_map={
                        "Positive (4-5 ⭐)": "#10b981",
                        "Negative (1-3 ⭐)": "#ef4444"
                    },
                    title=f"Sentiment Split (Total: {total_feedbacks} Responses)"
                )

                fig.update_traces(textposition='outside', marker_line_width=0)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color="#64748b",
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=20),
                    yaxis_title="Number of Feedbacks",
                    xaxis_title=""
                )
        
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

  
                st.markdown("### 📑 Recent Feedback Details")
                st.dataframe(
                    feedback_df.sort_values(by="timestamp", ascending=False),
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Date", format="D MMM, hh:mm a"),
                        "rating": st.column_config.NumberColumn("Rating", format="%d ⭐"),
                        "input_text": "Prompt",
                        "comment": "User Remarks"
                    },
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No feedback data detected in feedback.csv.")

        elif st.session_state.active_tab == "About us" and st.session_state.user_role == "user":    
            _, about_col, _ = st.columns([1, 4, 1])
        
            with about_col:

                st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 25px;">
                        <h2 style="color:{text_col}; margin-top:0;">🚀 Our Mission</h2>
                        <p style="font-size:18px; line-height:1.6; color:#64748b;">
                            At <b>Multimodal.ai</b>, we believe that the next frontier of artificial intelligence isn't just about larger models, 
                            but about <b>smarter collaboration</b>. Our mission is to bridge the gap between complex unstructured data 
                            and actionable intelligence through high-fidelity multi-agent orchestration.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 25px;">
                        <h3 style="color:{text_col};">🧪 How We Innovate</h3>
                        <p style="color:#64748b;">
                            Unlike standard LLM implementations, our system utilizes a <b>Post-Trained Multimodal Architecture</b>. 
                            By specifically fine-tuning models on domain-specific datasets, we achieve higher reasoning capabilities 
                            while significantly reducing the risk of hallucinations.
                        </p>
                        <ul style="color:#64748b; line-height:1.8;">
                            <li><b>Autonomous Routing:</b> Intelligence that knows exactly which specialized agent to deploy for your task.</li>
                            <li><b>Cross-Modal Verification:</b> Systems that cross-reference text, image, and data feeds for 100% accuracy.</li>
                            <li><b>Enterprise Security:</b> Built with a "Security-First" mindset, ensuring PII redaction and local execution options.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 25px;">
                        <h3 style="color:{text_col};">🏢 Our Foundation</h3>
                        <p style="color:#64748b;">
                            Founded by a team of AI researchers and software architects, Multimodal.ai was born out of the 
                            necessity to handle the explosion of multi-format data in modern enterprises. From object detection 
                            to document analysis and image and audio generation, our agents are designed to think like experts.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="glass-card" style="background:rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2);">
                        <h3 style="color:{text_col}; margin-top:0;">📧 Get In Touch</h3>
                        <div style="display: flex; gap: 40px; margin-top: 20px;">
                            <div>
                                <p style="margin:0; font-weight:600; color:{text_col};">Support</p>
                                <p style="color:#64748b;">support@multimodal.ai</p>
                            </div>
                            <div>
                                <p style="margin:0; font-weight:600; color:{text_col};">Partnerships</p>
                                <p style="color:#64748b;">partners@multimodal.ai</p>
                            </div>
                            <div>
                                <p style="margin:0; font-weight:600; color:{text_col};">Headquarters</p>
                                <p style="color:#64748b;">Digital Infrastructure, India</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)


            st.markdown("<br><br><br>", unsafe_allow_html=True) # Space before fixed footer

        elif st.session_state.active_tab == "Users" and st.session_state.user_role == "admin":
            st.markdown("### 👥 Registered Users")
            user_data = pd.read_csv(USER_DB)
    
            col_a, col_b = st.columns(2)
            col_a.metric("Total Users", len(user_data))
            col_b.metric("Latest Signup", user_data.iloc[-1]['username'])
    

            st.dataframe(user_data, use_container_width=True)
    
            if st.button("Export User List"):
                st.download_button("Download CSV", user_data.to_csv(), "users_export.csv")

        else:
            st.session_state.active_tab = "Dashboard"
            st.warning("Access Restricted: Redirecting to Dashboard.")

# SIDEBAR

    with st.sidebar:
        st.markdown("### 🧠 Active Agents")
    
        st.markdown("""
            <div class="agent-badge">Summarizer <span class="icon">✅</span></div>
            <div class="agent-badge">Vision System <span class="icon">✅</span></div>
            <div class="agent-badge">Audio Gen <span class="icon">✅</span></div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 AI Performance")
    
        FILE = os.path.join(os.path.dirname(__file__), "../backend/evaluation/metrics.csv")

        if os.path.exists(FILE):
            df = pd.read_csv(FILE)
            df["hallucination"] = df["hallucination"].astype(str).str.lower() == "true"

            if len(df) > 0:
                st.metric("System Accuracy", f"{round(df['confidence'].mean() *100, 2)}%")
                st.metric("Response Latency", f"{round(df['latency'].mean(), 2)}s")
                st.metric("Hallucination Rate", f"{round(df['hallucination'].mean() * 100, 1)}%")
            else:
                st.caption("Run agents to populate telemetry.")
        else:
            st.caption("No performance metrics yet.")

# FOOTER

    st.markdown(f"""
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background: {header_bg}; 
                text-align: center; padding: 10px; border-top: 1px solid {border_col}; font-size: 16px; color: #64748b;">
            <b>MULTIMODAL.AI</b> | © 2026 ALL RIGHTS RESERVED | Enterprise AI System • Build v2.4.0
        </div>
    """, unsafe_allow_html=True)


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        font-family: 'Inter', sans-serif;
    }}

    /* Header Container - Similar to image_a27802.png */
    .custom-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 2rem;
  
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
    }}

    .logo-section {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}

    .logo-text {{
        font-weight: 700;
        font-size: 1.2rem;
    }}

    /* Main Title Styling */
    .main-title {{
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #3b82f6, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 40px;
        margin-bottom: 4px;
    }}

    /* Button Styling */
    .stButton > button {{
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
</style>
""", unsafe_allow_html=True)

