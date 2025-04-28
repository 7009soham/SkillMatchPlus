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
        gdown.download(
            "https://drive.google.com/uc?id=1EPxqmQXd22QEA3shTkyDQTJgEcSWbx_1",
            embeddings_path,
            quiet=False
        )
    if not os.path.exists(faiss_index_path):
        gdown.download(
            "https://drive.google.com/uc?id=1lfnshv_eCvviasRLX6bYwwWQgk06fq7y",
            faiss_index_path,
            quiet=False
        )
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
    cursor.execute(
        "INSERT INTO users (Name, DOB, City, Profile_Text) VALUES (?, ?, ?, ?)",
        (name, dob, city, profile_text)
    )
    conn.commit()

# --- Custom CSS ---
st.markdown("""<style>/* (your existing CSS unchanged) */</style>""", unsafe_allow_html=True)

# --- Title and Ticker ---
st.markdown('<div class="big-title">🚀 SkillMatch+ | Connect Futuristically</div>', unsafe_allow_html=True)
st.markdown('<div class="moving-text">⚡ More Features Coming Soon! Get Ready for Futuristic Friendships ⚡</div>', unsafe_allow_html=True)
st.markdown("---")

# --- User Input Section ---
st.subheader("🧑‍🎓 Create Your Profile")
name = st.text_input("Enter Your Name")
dob = st.date_input(
    "Enter Your Date of Birth (DOB)",
    min_value=datetime(1950, 1, 1),
    max_value=datetime.now(),
    format="YYYY-MM-DD"
)
city = st.text_input("Enter Your City")

dataset = load_users()
all_interests = sorted(set(" ".join(dataset['Profile_Text']).split()))
selected_interests = st.multiselect("Choose Your Interests", options=all_interests)

if st.button("📝 Create My Profile"):
    if not name or not selected_interests:
        st.warning("Please fill all fields and select at least one interest.")
    else:
        profile_txt = " ".join(selected_interests)
        insert_user(name, dob.strftime("%Y-%m-%d"), city, profile_txt)
        st.success("✅ Profile Created Successfully!")

st.markdown("---")

# --- Match Recommendation Section ---
st.subheader("🔎 Find Matching Friends")
top_n = st.slider("Select Number of Recommendations", 5, 50, value=5)

# Prepare session for toggles
if 'show_mutuals' not in st.session_state:
    st.session_state.show_mutuals = {}

if st.button("✨ Find My Matches"):
    if not selected_interests:
        st.warning("⚡ Please select at least one interest to proceed.")
    else:
        st.success(f"Welcome {name or 'User'}! Finding your top {top_n} matches...")

        txt = " ".join(selected_interests)
        emb = encoder.encode(txt)
        emb = np.array([emb]).astype('float32')
        distances, indices = index.search(emb, top_n + 1)

        st.markdown("---")
        st.subheader(f"🎉 Top {top_n} Recommended Friends")
        st.markdown('<div class="cards-wrapper">', unsafe_allow_html=True)

        for rank, idx in enumerate(indices[0][1:top_n+1], start=1):
            if idx >= len(dataset):
                continue
            u = dataset.iloc[idx]
            featured = (rank == 1)
            card_cls = "profile-card featured" if featured else "profile-card"

            dob_str = u['DOB']
            try:
                dob_dt = datetime.strptime(dob_str, "%Y-%m-%d")
                age = (datetime.now() - dob_dt).days // 365
            except:
                age = "Unknown"

            pos = np.where(indices[0] == idx)[0][0]
            sim = round((1 - distances[0][pos]) * 100, 2)

            st.markdown(f"""
                <div class="{card_cls}">
                  <h4>👤 {u['Name']} from {u['City'] or 'Unknown'}</h4>
                  <p><strong>Age:</strong> {age} years</p>
                  <p><strong>Interests:</strong> 🌟 {u['Profile_Text']}</p>
                  <p><strong>Similarity:</strong> 🔥 {sim}%</p>
                  <button class="send-btn">🤝 Send Friend Request</button>
            """, unsafe_allow_html=True)

            # 🔥 Correct mutuals inside card using toggle
            key = f"mutuals_{idx}"
            show_mutual = st.toggle(f"🔎 View Mutual Interests for {u['Name']}", key=key)

            if show_mutual:
                current_interests = set(selected_interests)
                user_interests = set(u['Profile_Text'].split())

                mutual_interests = current_interests.intersection(user_interests)
                if mutual_interests:
                    st.markdown("<p><strong>🤝 Mutual Interests:</strong></p>", unsafe_allow_html=True)
                    for mutual in mutual_interests:
                        st.markdown(f"<p>• {mutual}</p>", unsafe_allow_html=True)
                else:
                    st.info("No mutual interests found.")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# --- Analytics & Visualizations Section ---
st.markdown("---")
st.subheader("📊 Community Insights & Trends")

st.markdown("### 🌟 Most Popular Interests")
if len(dataset) > 0:
    all_words = " ".join(dataset['Profile_Text']).split()
    interests_series = pd.Series(all_words).value_counts().head(20)
    st.bar_chart(interests_series)

st.markdown("### 🎂 Age Distribution of Users")
if len(dataset) > 0:
    def calculate_age(dob_str):
        try:
            dob_dt = datetime.strptime(dob_str, "%Y-%m-%d")
            return (datetime.now() - dob_dt).days // 365
        except:
            return None

    dataset['Age'] = dataset['DOB'].apply(calculate_age)
    age_distribution = dataset['Age'].dropna()

    st.bar_chart(age_distribution.value_counts().sort_index())

if 'distances' in locals():
    st.markdown("### 🔥 Similarity Score Distribution (Your Recommendations)")
    sim_scores = [(1 - d) * 100 for d in distances[0][1:top_n+1]]
    sim_df = pd.DataFrame({
        "Friend Rank": list(range(1, len(sim_scores)+1)),
        "Similarity (%)": sim_scores
    })
    st.line_chart(sim_df.set_index("Friend Rank"))

st.markdown("### 🏙️ Top Cities by User Count")
if len(dataset) > 0:
    city_counts = dataset['City'].value_counts().head(10)
    st.bar_chart(city_counts)

st.markdown("### 📈 Quick Summary")
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Users", len(dataset))
col2.metric("🎂 Avg. Age", round(age_distribution.mean(), 1) if not age_distribution.empty else "N/A")
col3.metric("🌆 Unique Cities", dataset['City'].nunique())

st.caption("Made with ❤️ | SkillMatch+ Cyberpunk MacOS Edition 🚀")
