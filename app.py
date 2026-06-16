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
# ULTRA MODERN FRONTEND CSS
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* =====================================================
BACKGROUND
===================================================== */

.stApp {

    background:
    linear-gradient(
        rgba(2,6,23,0.90),
        rgba(15,23,42,0.92)
    ),
    url("https://images.unsplash.com/photo-1509062522246-3755977927d7");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;

    overflow-x: hidden;
}

/* =====================================================
HIDE STREAMLIT
===================================================== */

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* =====================================================
TITLE
===================================================== */

.main-title {

    text-align:center;

    font-size:78px;

    font-weight:800;

    margin-top:20px;

    letter-spacing:1px;

    background: linear-gradient(
        to right,
        #38BDF8,
        #818CF8,
        #A855F7,
        #EC4899
    );

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

    text-shadow:0 0 40px rgba(255,255,255,0.15);

    animation: glow 4s ease infinite;
}

@keyframes glow {

    0% {
        filter: brightness(1);
    }

    50% {
        filter: brightness(1.2);
    }

    100% {
        filter: brightness(1);
    }
}

/* =====================================================
SUBTITLE
===================================================== */

.sub-title {

    text-align:center;

    font-size:24px;

    color:#E2E8F0;

    margin-bottom:40px;

    font-weight:400;

    letter-spacing:0.5px;
}

/* =====================================================
GLASS CARD
===================================================== */

.glass-card {

    background: rgba(15,23,42,0.72);

    border-radius:30px;

    padding:40px;

    backdrop-filter: blur(22px);

    border:1px solid rgba(255,255,255,0.10);

    box-shadow:
    0 10px 40px rgba(0,0,0,0.45),
    0 0 30px rgba(99,102,241,0.12);

    transition:0.4s ease;
}

.glass-card:hover {

    transform: translateY(-3px);

    box-shadow:
    0 20px 50px rgba(0,0,0,0.55),
    0 0 35px rgba(99,102,241,0.25);
}

/* =====================================================
INPUT BOX
===================================================== */

.stTextInput > div > div > input {

    background: rgba(255,255,255,0.96);

    border:2px solid transparent;

    border-radius:20px;

    padding:18px;

    color:black;

    font-size:17px;

    font-weight:500;

    transition:0.3s ease;

    box-shadow:0 5px 18px rgba(0,0,0,0.15);
}

.stTextInput > div > div > input:focus {

    border:2px solid #8B5CF6;

    box-shadow:
    0 0 20px rgba(139,92,246,0.45);
}

/* PLACEHOLDER */

input::placeholder {

    color:#475569 !important;
}

/* =====================================================
BUTTON
===================================================== */

.stButton > button {

    width:100%;

    border:none;

    border-radius:20px;

    padding:16px;

    background: linear-gradient(
        90deg,
        #4F46E5,
        #7C3AED,
        #A855F7
    );

    color:white;

    font-size:19px;

    font-weight:700;

    transition:0.35s ease;

    box-shadow:0 10px 25px rgba(124,58,237,0.35);
}

.stButton > button:hover {

    transform: translateY(-3px) scale(1.01);

    box-shadow:
    0 15px 35px rgba(124,58,237,0.5);

    background: linear-gradient(
        90deg,
        #6366F1,
        #8B5CF6,
        #D946EF
    );
}

/* =====================================================
USER CHAT
===================================================== */

.chat-user {

    background: linear-gradient(
        90deg,
        #2563EB,
        #3B82F6
    );

    padding:18px;

    border-radius:18px;

    margin:14px 0;

    color:white;

    font-size:17px;

    font-weight:500;

    box-shadow:0 8px 20px rgba(37,99,235,0.35);
}

/* =====================================================
BOT CHAT
===================================================== */

.chat-bot {

    background: rgba(255,255,255,0.08);

    border-left:5px solid #A855F7;

    padding:22px;

    border-radius:18px;

    margin:14px 0;

    color:#F8FAFC;

    line-height:1.9;

    font-size:17px;

    box-shadow:0 8px 20px rgba(0,0,0,0.25);
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #111827,
        #0F172A
    );

    border-right:1px solid rgba(255,255,255,0.08);
}

/* SIDEBAR TEXT */

section[data-testid="stSidebar"] * {

    color:white !important;
}

/* =====================================================
IMAGES
===================================================== */

img {

    border-radius:22px !important;

    box-shadow:
    0 10px 30px rgba(0,0,0,0.45);

    transition:0.4s ease;
}

img:hover {

    transform: scale(1.03);

    box-shadow:
    0 15px 40px rgba(0,0,0,0.55);
}

/* =====================================================
FLOATING AI BOT
===================================================== */

.avatar {

    position:fixed;

    bottom:25px;

    right:25px;

    width:90px;

    height:90px;

    border-radius:50%;

    background: linear-gradient(
        135deg,
        #4F46E5,
        #8B5CF6,
        #EC4899
    );

    display:flex;

    justify-content:center;

    align-items:center;

    font-size:42px;

    box-shadow:
    0 0 35px rgba(139,92,246,0.8);

    animation: float 3s ease-in-out infinite;

    z-index:999;
}

@keyframes float {

    0% {
        transform:translateY(0px);
    }

    50% {
        transform:translateY(-12px);
    }

    100% {
        transform:translateY(0px);
    }
}

/* =====================================================
FOOTER
===================================================== */

.footer {

    text-align:center;

    color:#CBD5E1;

    margin-top:50px;

    font-size:15px;

    letter-spacing:0.5px;
}

/* =====================================================
SCROLLBAR
===================================================== */

::-webkit-scrollbar {

    width:10px;
}

::-webkit-scrollbar-track {

    background:#0F172A;
}

::-webkit-scrollbar-thumb {

    background:linear-gradient(
        #6366F1,
        #A855F7
    );

    border-radius:20px;
}

/* =====================================================
MOBILE
===================================================== */

@media(max-width:768px){

.main-title{
    font-size:48px;
}

.sub-title{
    font-size:17px;
}

.glass-card{
    padding:22px;
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
    "color": {"value": "#8B5CF6"},
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
# CHUNKING
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
# FAISS
# =====================================================

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🤖 AI Features")

st.sidebar.markdown("""
✅ Smart AI Chatbot  
✅ Typing Animation  
✅ Chat History  
✅ Gemini AI  
✅ Particle Effects  
✅ Modern UI  
✅ Responsive Design  
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

            time.sleep(0.002)

# =====================================================
# CLOSE CARD
# =====================================================

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FLOATING AVATAR
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
