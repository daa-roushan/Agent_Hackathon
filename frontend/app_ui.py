import streamlit as st
import requests
import json
import time

# ========== CONFIG ==========
st.set_page_config(
    page_title="Accordion Hackathon Team 14",
    page_icon="🤖",
    layout="centered",
)

url = "http://127.0.0.1:5000/query"
headers = {'Content-Type': 'application/json'}

# ========== SESSION STATE ==========
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "result" not in st.session_state:
    st.session_state.result = ""

# ========== STYLING (Purple Theme) ==========
st.markdown(
    """
    <style>
    /* ===== Global Background Fix (removes white top bar) ===== */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background: linear-gradient(135deg, #4B0082 10%, #800080 50%, #8A2BE2 90%) !important;
        color: white !important;
        height: 100%;
        margin: 0;
        padding: 0;
    }

    /* Remove any white background from top container */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2D0B52 !important;
        color: white !important;
    }

    /* Main Title */
    .big-font {
        font-size: 28px !important;
        color: #E0BBFF !important;
        text-align: center;
        font-weight: bold;
        text-shadow: 1px 1px 2px #3A0CA3;
    }

    /* Textarea */
    textarea {
        background-color: #F5F3FF !important;
        color: #000000 !important;
        border: 1px solid #9B5DE5 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #9B5DE5 !important;
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-weight: 600;
        transition: 0.2s;
    }
    div.stButton > button:hover {
        background-color: #7B2CBF !important;
        transform: scale(1.03);
    }

    /* Footer */
    footer, .reportview-container .main footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ========== SIDEBAR ==========
with st.sidebar:
    st.image(
        "https://cdn3.iconfinder.com/data/icons/artificial-intelligence-1-2-1/1024/Artificial_intelligence-10-1024.png",
        width=100
    )
    st.title("About 🤖")
    st.write(
        """
        **Accordion Hackathon ChatBot**  
        Type your question and get context-aware answers!
        """
    )
    st.markdown("---")
    st.write("💡 *Tip:* Try asking store-specific questions!")

    st.markdown("---")
    st.markdown("### 👥 Team 14 Members")
    st.write(
        """
        - **Manish Jha**  
        - **Roushan Gupta** 
        - **Neeraj Yadav**  
        - **Aditya Kadiyala**
        """
    )

# ========== MAIN AREA ==========
st.markdown('<p class="big-font">Store Information AI Chat Assistant</p>', unsafe_allow_html=True)
st.write("Ask anything, and I’ll fetch insights for you! 🚀")

st.markdown("### 💬 Ask your question below:")
st.session_state.user_query = st.text_area(
    "",
    value=st.session_state.user_query,
    placeholder="e.g., Where is Store 1 located?",
    height=120,
    key="query_input"
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Submit", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

# ========== RESPONSE AREA ==========
if submit:
    if not st.session_state.user_query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        with st.spinner("🔎 Thinking... Please wait"):
            payload = json.dumps({"query": st.session_state.user_query})
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                st.session_state.result = data.get("response", "No response returned from backend.")
                
                time.sleep(0.8)
                st.success("✅ Response received!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")

if st.session_state.result:
    st.markdown("### 🧠 ChatBot Response:")
    st.text_area("Result", value=st.session_state.result, height=250)

# ========== CLEAR BUTTON ==========
if clear:
    st.session_state.user_query = ""
    st.session_state.result = ""
    st.rerun()

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #E0BBFF;'>
        © Team 14 — Accordion Hackathon 2025
    </div>
    """,
    unsafe_allow_html=True
)
