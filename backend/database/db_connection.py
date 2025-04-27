import os
import sqlite3
import numpy as np
import joblib
import faiss
from flask_pymongo import PyMongo

# Initialize Mongo (still optional for future use)
mongo = PyMongo()

# 📦 Correct Paths (relative to backend folder)
base_path = os.path.dirname(os.path.abspath(__file__))  # backend/database
backend_path = os.path.abspath(os.path.join(base_path, ".."))

# 📂 Load SQLite database
db_path = os.path.join(base_path, "skillmatch.db")
sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
sqlite_cursor = sqlite_conn.cursor()

# 📂 Load embeddings
embeddings_path = os.path.join(backend_path, "embeddings.npy")
embeddings = np.load(embeddings_path)

# 📂 Load FAISS index
faiss_index_path = os.path.join(backend_path, "faiss.index")
faiss_index = faiss.read_index(faiss_index_path)

# 📂 Load friendship model (optional if you're using friendship strength feature)
friendship_model_path = os.path.join(backend_path, "models", "friendship_model.pkl")
friendship_model = joblib.load(friendship_model_path)

# --- Helper functions for SQLite Access ---

def fetch_all_users():
    query = "SELECT UserID, Name, City, DOB, Profile_Text FROM users"
    sqlite_cursor.execute(query)
    rows = sqlite_cursor.fetchall()
    return rows

def fetch_user_by_id(user_id):
    query = "SELECT UserID, Name, City, DOB, Profile_Text FROM users WHERE UserID = ?"
    sqlite_cursor.execute(query, (user_id,))
    row = sqlite_cursor.fetchone()
    return row
