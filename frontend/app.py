import streamlit as st
import numpy as np
import pandas as pd
import faiss
import os
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Streamlit Config ---
st.set_page_config(
    page_title="SkillMatch+ | Find Your Tribe",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Encoder ---
@st.cache_resource()
def load_encoder():
    return SentenceTransformer('hkunlp/instructor-xl')

encoder = load_encoder()

# --- Paths ---
base_path = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.join(base_path, "..", "backend", "database", "skillmatch.db")
embeddings_path = os.path.join(base_path, "embeddings.npy")
faiss_index_path = os.path.join(base_path, "faiss.index")

# --- Load Embeddings and Index ---
@st.cache_resource()
def load_embeddings_and_index():
    embeddings = np.load(embeddings_path, allow_pickle=True)
    index = faiss.read_index(faiss_index_path)
    return embeddings, index

embeddings, index = load_embeddings_and_index()

# --- Connect to SQLite ---
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

def load_users():
    cursor.execute("SELECT UserID, Name, City, DOB, Profile_Text FROM users")
    data = cursor.fetchall()
    columns = ['UserID', 'Name', 'City', 'DOB', 'Profile_Text']
    return pd.DataFrame(data, columns=columns)

# --- Custom CSS for Cyberpunk MacOS Feel ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #eee;
        }
        .big-title {
            font-size: 3rem;
            font-weight: 900;
            background: linear-gradient(to right, #ff512f, #dd2476);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stTextInput > div > div > input {
            background-color: #222 !important;
            color: white !important;
            border: 1px solid #555;
            border-radius: 10px;
        }
        .stButton > button {
            background: linear-gradient(to right, #ff4b2b, #ff416c);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: bold;
            padding: 0.6rem 1.5rem;
        }
        .profile-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
        }
    </style>
""", unsafe_allow_html=True)

# --- UI Title ---
st.markdown('<div class="big-title">🌟 SkillMatch+ | Find Your Tribe</div>', unsafe_allow_html=True)
st.markdown("---")

# --- Load Users ---
dataset = load_users()

# --- Form ---
st.header("👤 Create Your Profile")
with st.form("user_profile_form"):
    name = st.text_input("Enter your full name")
    dob = st.date_input("Enter your Date of Birth")
    city = st.text_input("Enter your City")

    available_interests = sorted(list(set(' '.join(dataset['Profile_Text']).split())))
    selected_interests = st.multiselect("Select Your Interests", options=available_interests)

    submit_profile = st.form_submit_button("🚀 Create My Profile")

# --- Insert New User ---
if submit_profile:
    if name and city and selected_interests:
        new_profile_text = ' '.join(selected_interests)
        cursor.execute("INSERT INTO users (Name, City, DOB, Profile_Text) VALUES (?, ?, ?, ?)",
                       (name, city, dob.strftime('%Y-%m-%d'), new_profile_text))
        conn.commit()
        st.success("✅ Your profile has been created successfully!")
        dataset = load_users()
    else:
        st.error("⚠️ Please fill all fields.")

st.markdown("---")

# --- Recommendation ---
st.header("🔎 Find Matching Friends")

top_n = st.slider("Select number of recommendations", 5, 50, value=5)

if st.button("✨ Show My Matches"):
    if selected_interests:
        st.success("🔍 Searching for best matches...")

        user_embedding = encoder.encode(' '.join(selected_interests))
        user_embedding = np.array([user_embedding]).astype('float32')

        distances, indices = index.search(user_embedding, top_n + 1)

        st.markdown("---")
        st.subheader(f"🎯 Top {top_n} Matching Profiles")

        for idx in indices[0][1:]:
            if idx < len(dataset):
                matched_user = dataset.iloc[idx]
                user_dob = matched_user.get('DOB')
                age = None
                if pd.notnull(user_dob):
                    try:
                        birth_date = datetime.strptime(user_dob, '%Y-%m-%d')
                        today = datetime.today()
                        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                    except:
                        age = 'Unknown'

                with st.container():
                    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                    st.markdown(f"### 👤 {matched_user['Name']} from {matched_user.get('City', 'Unknown')}")
                    st.write(f"**Age:** {age if age else 'Not Available'}")
                    st.write(f"**Interests:** {matched_user['Profile_Text']}")
                    sim_score = round((1 - distances[0][np.where(indices[0] == idx)[0][0]]) * 100, 2)
                    st.write(f"**Semantic Match Score:** 🌟 {sim_score}%")
                    st.button(f"🤝 Connect with {matched_user['Name']}", key=f"connect_{idx}")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚡ Please select interests to get recommendations!")

st.markdown("---")
st.caption("Made with ❤️ SkillMatch+ | 2025 | Futuristic Edition 🚀")
