# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import os
import sqlite3
import gdown
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Always first Streamlit command ---
st.set_page_config(
    page_title="SkillMatch+ | Your Future Connections",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Encoder with Cache ---
@st.cache_resource()
def load_encoder():
    return SentenceTransformer('hkunlp/instructor-xl')

encoder = load_encoder()

# --- Paths ---
base_path = os.path.dirname(os.path.abspath(__file__))  # frontend/
database_path = os.path.join(base_path, "..", "backend", "database", "skillmatch.db")
embeddings_path = os.path.join(base_path, "embeddings.npy")
faiss_index_path = os.path.join(base_path, "faiss.index")

# --- Download embeddings and index if missing ---
@st.cache_resource()
def load_embeddings_and_index():
    if not os.path.exists(embeddings_path):
        gdown.download("https://drive.google.com/uc?id=1EPxqmQXd22QEA3shTkyDQTJgEcSWbx_1", embeddings_path, quiet=False)
    if not os.path.exists(faiss_index_path):
        gdown.download("https://drive.google.com/uc?id=1lfnshv_eCvviasRLX6bYwwWQgk06fq7y", faiss_index_path, quiet=False)
    
    embeddings = np.load(embeddings_path)
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

# --- Custom CSS for Dark Futuristic UI ---
st.markdown("""
    <style>
        .big-title {
            font-size: 42px;
            font-weight: 700;
            background: linear-gradient(to right, #ff4b1f, #1fddff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .profile-card {
            background-color: #1f1f1f;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            box-shadow: 0px 0px 12px rgba(255,255,255,0.1);
        }
        .send-btn {
            background-color: #ff4b1f;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- UI: Title ---
st.markdown('<div class="big-title">🚀 SkillMatch+ | Your Future Connections</div>', unsafe_allow_html=True)
st.markdown("---")

# --- UI: User Input ---
st.header("🧑‍🎓 Create Your Profile")

name = st.text_input("Enter Your Name")
age = st.number_input("Enter Your Age", min_value=10, max_value=100, step=1)
city = st.text_input("Enter Your City")

# Load users from database
dataset = load_users()

available_interests = sorted(list(set(' '.join(dataset['Profile_Text']).split(' '))))
selected_interests = st.multiselect("Choose Your Interests", options=available_interests)

# --- UI: Match Recommendation ---
st.header("🔎 Find Matching Friends")

top_n = st.slider("Select Number of Recommendations", 5, 50, value=5)

if st.button("✨ Find My Matches"):
    if not selected_interests:
        st.warning("⚡ Please select at least one interest!")
    else:
        st.success(f"Welcome {name or 'User'}! Finding your top {top_n} matches...")

        # --- Create New Embedding from selected interests ---
        new_profile_text = ' '.join(selected_interests)
        user_embedding = encoder.encode(new_profile_text)
        user_embedding = np.array([user_embedding]).astype('float32')

        # --- FAISS Search ---
        distances, indices = index.search(user_embedding, top_n + 1)  # +1 because first might be itself

        st.markdown("---")
        st.subheader(f"🎉 Top {top_n} Recommended Friends")

        # --- Show Recommendations ---
        for idx in indices[0][1:]:  # Skip 0th index (new query user itself)
            matched_user = dataset.iloc[idx]
            with st.container():
                st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                st.markdown(f"### 👤 {matched_user['Name']} from {matched_user.get('City', 'Unknown')}")
                dob = matched_user.get('DOB', 'Not available')
                st.write(f"**DOB:** {dob if pd.notnull(dob) else 'Not available'}")
                st.write(f"**Interests:** 🌟 {matched_user['Profile_Text']}")
                sim_score = round((1 - distances[0][np.where(indices[0] == idx)[0][0]]) * 100, 2)
                st.write(f"**Semantic Similarity:** 🔥 {sim_score}%")
                send_request_btn = st.button(f"🤝 Send Friend Request to {matched_user['Name']}", key=f"send_request_{idx}")
                if send_request_btn:
                    st.success(f"✅ Friend Request Sent to {matched_user['Name']}!")
                st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ by SkillMatch+ Team | 2025 Edition 🚀")
