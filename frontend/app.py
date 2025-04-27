# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import os
import sqlite3
import random
import gdown
from datetime import datetime
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

# --- Streamlit Page Settings ---
st.set_page_config(page_title="SkillMatch+ Cyberpunk", layout="wide", initial_sidebar_state="expanded")

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

# --- Download embeddings and index if missing ---
@st.cache_resource()
def load_embeddings_and_index():
    if not os.path.exists(embeddings_path):
        gdown.download("https://drive.google.com/uc?id=1EPxqmQXd22QEA3shTkyDQTJgEcSWbx_1", embeddings_path, quiet=False)
    if not os.path.exists(faiss_index_path):
        gdown.download("https://drive.google.com/uc?id=1lfnshv_eCvviasRLX6bYwwWQgk06fq7y", faiss_index_path, quiet=False)
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

def insert_user(name, dob, city, profile_text):
    cursor.execute("INSERT INTO users (Name, DOB, City, Profile_Text) VALUES (?, ?, ?, ?)", (name, dob, city, profile_text))
    conn.commit()

# --- Custom CSS ---
st.markdown("""
<style>
body { background-color: #0d0d0d; }
.big-title {
    font-size: 48px;
    font-weight: 900;
    background: linear-gradient(90deg, #00ffe5, #ff00c8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.profile-scroll {
    overflow-x: scroll;
    white-space: nowrap;
    padding: 1rem;
    animation: slowmove 60s linear infinite;
}
.profile-scroll:hover {
    animation-play-state: paused;
}
.profile-card {
    display: inline-block;
    vertical-align: top;
    width: 300px;
    margin: 0 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 15px rgba(0,255,255,0.3);
    padding: 1.5rem;
}
.featured {
    border: 2px solid #ff00c8;
    box-shadow: 0 0 20px #ff00c8;
    position: relative;
}
.featured-badge {
    position: absolute;
    top: -10px;
    right: -10px;
    background: #ff00c8;
    color: white;
    padding: 3px 8px;
    font-size: 10px;
    border-radius: 10px;
    font-weight: bold;
}
.moving-text {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
    box-sizing: border-box;
    animation: marquee 20s linear infinite;
    font-weight: bold;
    color: #00ffe5;
    font-size: 18px;
    margin-bottom: 1rem;
}
.moving-text:hover {
    animation-play-state: paused;
}
@keyframes marquee {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}
@keyframes slowmove {
    0% { transform: translateX(0%); }
    100% { transform: translateX(-50%); }
}
</style>
""", unsafe_allow_html=True)

# --- UI Title ---
st.markdown('<div class="big-title">🚀 SkillMatch+ | Cyberpunk Connections</div>', unsafe_allow_html=True)

# --- Advertisement Text ---
st.markdown('<div class="moving-text">⚡ More Features Coming Soon! Welcome to the Future of Friendships ⚡</div>', unsafe_allow_html=True)

# --- User Profile Form ---
st.subheader("🧑‍🎓 Create Your Profile")

name = st.text_input("Enter Your Name")
dob = st.date_input("Enter Your Date of Birth (DOB)", min_value=datetime(1950, 1, 1), max_value=datetime.now())
city = st.text_input("Enter Your City")

dataset = load_users()
available_interests = sorted(list(set(' '.join(dataset['Profile_Text']).split(' '))))
selected_interests = st.multiselect("Choose Your Interests", options=available_interests)

if st.button("📝 Create My Profile"):
    if not name or not selected_interests:
        st.warning("Please fill all fields and select interests.")
    else:
        profile_text = ' '.join(selected_interests)
        insert_user(name, dob.strftime("%Y-%m-%d"), city, profile_text)
        st.success("Profile Created Successfully!")

# --- Match Recommendations ---
st.markdown("---")
st.subheader("🔎 Find Matching Friends")

top_n = st.slider("Select Number of Recommendations", 5, 50, value=5)

if st.button("✨ Find My Matches"):
    if not selected_interests:
        st.warning("⚡ Please select at least one interest!")
    else:
        st.success(f"Welcome {name or 'User'}! Finding your top {top_n} matches...")

        # --- Create New Embedding ---
        user_text = ' '.join(selected_interests)
        user_embedding = encoder.encode(user_text)
        user_embedding = np.array([user_embedding]).astype('float32')

        distances, indices = index.search(user_embedding, top_n + 1)

        featured_idx = random.choice(indices[0][1:])

        st.markdown("---")
        st.subheader(f"🎉 Top {top_n} Recommended Friends")
        st.markdown('<div class="profile-scroll">', unsafe_allow_html=True)

        for idx in indices[0][1:]:
            if idx < len(dataset):
                matched_user = dataset.iloc[idx]
                age = 'Unknown'
                if pd.notnull(matched_user.get('DOB')):
                    try:
                        dob_date = datetime.strptime(matched_user['DOB'], "%Y-%m-%d")
                        age = (datetime.now() - dob_date).days // 365
                    except:
                        pass

                is_featured = (idx == featured_idx)
                card_classes = "profile-card featured" if is_featured else "profile-card"

                st.markdown(f"""
                <div class="{card_classes}">
                    {"<div class='featured-badge'>⭐ FEATURED</div>" if is_featured else ""}
                    <h4>👤 {matched_user['Name']}</h4>
                    <p><strong>City:</strong> {matched_user.get('City', 'Unknown')}</p>
                    <p><strong>Age:</strong> {age} years</p>
                    <p><strong>Interests:</strong> {matched_user['Profile_Text']}</p>
                    <p><strong>Similarity:</strong> {round((1 - distances[0][np.where(indices[0]==idx)[0][0]])*100, 2)}%</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ by SkillMatch+ | Cyberpunk Edition 🚀")
