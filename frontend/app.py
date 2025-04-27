# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import faiss
import os
import sqlite3
import gdown
from datetime import datetime
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Always first Streamlit command ---
st.set_page_config(
    page_title="SkillMatch+ | Connect Futuristically",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Encoder with Cache ---
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
        body { background-color: #0f0f0f; }
        .big-title {
            font-size: 48px;
            font-weight: 900;
            background: linear-gradient(90deg, #00ffe5, #ff00c8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        .profile-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.8rem;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            margin-bottom: 1.5rem;
            box-shadow: 0px 0px 15px 3px rgba(0,255,255,0.2);
            transition: transform 0.5s ease;
        }
        .profile-card:hover {
            transform: scale(1.02);
        }
        .send-btn {
            background: linear-gradient(to right, #00ffe5, #ff00c8);
            color: black;
            font-weight: bold;
            border-radius: 12px;
            width: 100%;
        }
        .moving-text {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            animation: marquee 15s linear infinite;
            font-weight: bold;
            color: #ff00c8;
            background: transparent;
            padding: 0.5rem;
            font-size: 18px;
        }
        @keyframes marquee {
            0% { transform: translateX(100%); }
            100% { transform: translateX(-100%); }
        }
    </style>
""", unsafe_allow_html=True)

# --- UI: Title ---
st.markdown('<div class="big-title">🚀 SkillMatch+ | Connect Futuristically</div>', unsafe_allow_html=True)

# --- Moving Advertisement ---
st.markdown('<div class="moving-text">⚡ More Features Coming Soon! Get Ready for the Future of Connections. ⚡</div>', unsafe_allow_html=True)

st.markdown("---")

# --- UI: User Input ---
st.subheader("🧑‍🎓 Create Your Profile")

name = st.text_input("Enter Your Name")
dob = st.date_input("Enter Your Date of Birth (DOB)", min_value=datetime(1950, 1, 1), max_value=datetime.now(), format="YYYY-MM-DD")
city = st.text_input("Enter Your City")

# Load users from database
dataset = load_users()

available_interests = sorted(list(set(' '.join(dataset['Profile_Text']).split(' '))))
selected_interests = st.multiselect("Choose Your Interests", options=available_interests)

# --- Save New User to Database ---
if st.button("📝 Create My Profile"):
    if not name or not selected_interests:
        st.warning("Please fill all fields and select interests.")
    else:
        new_profile_text = ' '.join(selected_interests)
        insert_user(name, dob.strftime("%Y-%m-%d"), city, new_profile_text)
        st.success("Profile Created Successfully!")

st.markdown("---")

# --- UI: Match Recommendation ---
st.subheader("🔎 Find Matching Friends")

top_n = st.slider("Select Number of Recommendations", 5, 50, value=5)

if st.button("✨ Find My Matches"):
    if not selected_interests:
        st.warning("⚡ Please select at least one interest!")
    else:
        st.success(f"Welcome {name or 'User'}! Finding your top {top_n} matches...")

        # --- Create New Embedding ---
        new_profile_text = ' '.join(selected_interests)
        user_embedding = encoder.encode(new_profile_text)
        user_embedding = np.array([user_embedding]).astype('float32')

        # --- FAISS Search ---
        distances, indices = index.search(user_embedding, top_n + 1)

        st.markdown("---")
        st.subheader(f"🎉 Top {top_n} Recommended Friends")

        # --- Show Recommendations ---
        for idx in indices[0][1:]:
            if idx < len(dataset):
                matched_user = dataset.iloc[idx]
                with st.container():
                    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
                    st.markdown(f"### 👤 {matched_user['Name']} from {matched_user.get('City', 'Unknown')}")
                    
                    dob_str = matched_user.get('DOB', 'Unknown')
                    if pd.notnull(dob_str) and dob_str != 'Unknown':
                        try:
                            dob_date = datetime.strptime(dob_str, "%Y-%m-%d")
                            age_years = (datetime.now() - dob_date).days // 365
                        except Exception:
                            age_years = 'Unknown'
                    else:
                        age_years = 'Unknown'
                    
                    st.write(f"**Age:** {age_years} years")
                    st.write(f"**Interests:** 🌟 {matched_user['Profile_Text']}")
                    sim_score = round((1 - distances[0][np.where(indices[0] == idx)[0][0]]) * 100, 2)
                    st.write(f"**Semantic Similarity:** 🔥 {sim_score}%")
                    
                    # --- Buttons Section ---
                    cols = st.columns(2, gap="small")
                    with cols[0]:
                        if st.button(f"🤝 Send Friend Request to {matched_user['Name']}", key=f"send_{idx}"):
                            st.success(f"✅ Friend Request Sent to {matched_user['Name']}!")
                    with cols[1]:
                        if st.button(f"🔍 View Mutuals with {matched_user['Name']}", key=f"view_{idx}"):
                            user_interests = set(selected_interests)
                            matched_interests = set(matched_user['Profile_Text'].split())
                            mutual_interests = user_interests.intersection(matched_interests)
                            different_interests = matched_interests - user_interests

                            st.markdown(f"""
                                <div style="background: rgba(0, 255, 255, 0.05); padding: 1rem; margin-top: 10px; border-radius: 10px; backdrop-filter: blur(4px);">
                                    <h5>🤝 Mutual Interests:</h5>
                                    <p>{', '.join(mutual_interests) if mutual_interests else 'No mutual interests.'}</p>
                                    <h5>🎯 Other Interests:</h5>
                                    <p>{', '.join(different_interests) if different_interests else 'None'}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ | SkillMatch+ Cyberpunk Edition 🚀")
