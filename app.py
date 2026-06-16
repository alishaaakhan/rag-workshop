import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit.components.v1 as components
import time

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SISTec AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND */

.stApp {

    background:
    linear-gradient(
        rgba(15,23,42,0.92),
        rgba(15,23,42,0.92)
    ),
    url("https://images.unsplash.com/photo-1523050854058-8df90110c9f1");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;

    color: white;
}

/* HIDE STREAMLIT */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* MAIN TITLE */

.main-title {

    text-align:center;

    font-size:68px;

    font-weight:700;

    margin-top:10px;

    background: linear-gradient(
        to right,
        #38BDF8,
        #818CF8,
        #C084FC
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;
}

/* SUBTITLE */

.sub-title {

    text-align:center;

    color:#E2E8F0;

    font-size:22px;

    margin-bottom:35px;
}

/* GLASS CARD */

.glass-card {

    background: rgba(17,24,39,0.78);

    border-radius:25px;

    padding:35px;

    backdrop-filter: blur(18px);

    border:1px solid rgba(255,255,255,0.1);

    box-shadow:0 10px 35px rgba(0,0,0,0.45);
}

/* INPUT BOX */

.stTextInput > div > div > input {

    background: rgba(255,255,255,0.95);

    border:2px solid #6366F1;

    border-radius:18px;

    padding:16px;

    color:black;

    font-size:17px;

    font-weight:500;
}

/* PLACEHOLDER */

input::placeholder {

    color:#475569 !important;
}

/* BUTTON */

.stButton > button {

    width:100%;

    border:none;

    border-radius:18px;

    padding:15px;

    background: linear-gradient(
        90deg,
        #6366F1,
        #8B5CF6
    );

    color:white;

    font-size:18px;

    font-weight:600;

    transition:0.3s;
}

.stButton > button:hover {

    transform:translateY(-2px);

    box-shadow:0 10px 25px rgba(99,102,241,0.5);
}

/* CHAT USER */

.chat-user {

    background: linear-gradient(
        90deg,
        #2563EB,
        #3B82F6
    );

    padding:16px;

    border-radius:16px;

    margin:12px 0;

    color:white;

    font-size:17px;
}

/* CHAT BOT */

.chat-bot {

    background: rgba(255,255,255,0.08);

    border-left:5px solid #8B5CF6;

    padding:18px;

    border-radius:16px;

    margin:12px 0;

    color:#F8FAFC;

    line-height:1.8;

    font-size:17px;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {

    background:#111827;
}

/* SIDEBAR TEXT */

section[data-testid="stSidebar"] * {

    color:white !important;
}

/* IMAGES */

img {

    border-radius:18px !important;

    box-shadow:0 8px 25px rgba(0,0,0,0.35);
}

/* FLOATING AVATAR */

.avatar {

    position:fixed;

    bottom:20px;

    right:20px;

    width:85px;

    height:85px;

    border-radius:50%;

    background: linear-gradient(
        135deg,
        #6366F1,
        #8B5CF6
    );

    display:flex;

    justify-content:center;

    align-items:center;

    font-size:38px;

    box-shadow:0 0 30px rgba(99,102,241,0.7);

    animation: float 3s ease-in-out infinite;

    z-index:999;
}

@keyframes float {

    0% {transform:translateY(0px);}
    50% {transform:translateY(-10px);}
    100% {transform:translateY(0px);}
}

/* FOOTER */

.footer {

    text-align:center;

    color:#CBD5E1;

    margin-top:40px;

    font-size:15px;
}

/* MOBILE */

@media(max-width:768px){

.main-title{
    font-size:42px;
}

.sub-title{
    font-size:16px;
}

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# PARTICLE BACKGROUND
# =====================================================

particles_js = """
<div id="particles-js"></div>

<style>
#particles-js{
position:fixed;
width:100%;
height:100%;
z-index:-1;
top:0;
left:0;
}
</style>

<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>

<script>
particlesJS("particles-js", {
  "particles": {
    "number": {"value": 80},
    "color": {"value": "#6366F1"},
    "shape": {"type": "circle"},
    "opacity": {"value": 0.5},
    "size": {"value": 3},
    "move": {"enable": true, "speed": 2}
  }
});
</script>
"""

components.html(particles_js, height=0)

# =====================================================
# TITLE
# =====================================================

st.markdown("""
<div class='main-title'>
🎓 SISTec AI Assistant
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
Smart AI Assistant for Admissions, Fees, Placements, Hostel & Campus Information
</div>
""", unsafe_allow_html=True)

# =====================================================
# GEMINI API
# =====================================================

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("❌ Gemini API Key Missing")
    st.stop()

# =====================================================
# LOAD DATA
# =====================================================

with open("sistec_data.txt", "r", encoding="utf-8") as f:
    data = f.read()

# =====================================================
# TEXT CHUNKING
# =====================================================

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks

texts = chunk_text(data)

# =====================================================
# EMBEDDINGS
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model_emb = load_model()

embeddings = model_emb.encode(texts).astype("float32")

# =====================================================
# FAISS INDEX
# =====================================================

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🤖 Features")

st.sidebar.markdown("""
✅ AI Chatbot  
✅ Voice Assistant UI  
✅ Typing Animation  
✅ Chat History  
✅ Particle Background  
✅ Smart Search  
✅ Gemini AI  
✅ Responsive UI  
""")

# =====================================================
# CAMPUS IMAGES
# =====================================================

st.markdown("## 🏫 SISTec Campus")

st.image(
    [
        "https://images.unsplash.com/photo-1562774053-701939374585",
        "https://images.unsplash.com/photo-1498243691581-b145c3f54a5a",
        "https://images.unsplash.com/photo-1541339907198-e08756dedf3f",
        "https://images.unsplash.com/photo-1519389950473-47ba0277781c"
    ],
    caption=[
        "Modern Campus",
        "Advanced Learning Environment",
        "Student Life",
        "Innovation Labs"
    ],
    width=260
)

# =====================================================
# MAIN CARD
# =====================================================

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

question = st.text_input(
    "💬 Ask Your Question",
    placeholder="Example: What is the hostel fee?"
)

voice_btn = st.button("🎤 Voice Assistant")

if voice_btn:
    st.info("Voice assistant feature can be added using speech recognition.")

# =====================================================
# GENERATE ANSWER
# =====================================================

if st.button("🚀 Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question")

    else:

        st.session_state.messages.append(("user", question))

        q_emb = model_emb.encode([question]).astype("float32")

        distances, indices = index.search(q_emb, k=4)

        context = "\n\n".join([texts[i] for i in indices[0]])

        prompt = f"""
You are SISTec AI Assistant.

Answer only from provided context.

If information is unavailable say:
'Sorry, I do not have information about that.'

Context:
{context}

Question:
{question}

Answer:
"""

        llm = genai.GenerativeModel("gemini-2.5-flash-lite")

        with st.spinner("🤖 AI Thinking..."):

            response = llm.generate_content(prompt)

            answer = response.text

        st.session_state.messages.append(("bot", answer))

# =====================================================
# CHAT HISTORY
# =====================================================

for role, msg in st.session_state.messages:

    if role == "user":

        st.markdown(
            f"<div class='chat-user'>👤 {msg}</div>",
            unsafe_allow_html=True
        )

    else:

        placeholder = st.empty()

        typed = ""

        for char in msg:

            typed += char

            placeholder.markdown(
                f"<div class='chat-bot'>🤖 {typed}</div>",
                unsafe_allow_html=True
            )

            time.sleep(0.003)

# =====================================================
# CLOSE CARD
# =====================================================

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FLOATING BOT
# =====================================================

st.markdown("""
<div class='avatar'>
🤖
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOTTIE ANIMATION
# =====================================================

components.html("""
<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>

<lottie-player
src="https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json"
background="transparent"
speed="1"
style="width:250px;height:250px;margin:auto;"
loop
autoplay>
</lottie-player>
""", height=260)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class='footer'>
Made by Alisha Khan ❤️
</div>
""", unsafe_allow_html=True)
