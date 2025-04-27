from database.db_connection import embeddings, faiss_index, sqlite_conn
import numpy as np
import sqlite3

def fetch_user_by_id(user_id):
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT UserID, Name, City, DOB, Profile_Text FROM users WHERE UserID = ?", (user_id,))
    return cursor.fetchone()

def fetch_all_users():
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT UserID, Name, City, DOB, Profile_Text FROM users")
    return cursor.fetchall()

def get_top_matches(user_id, top_n=5):
    if user_id >= len(embeddings):
        return []

    user_embedding = np.array([embeddings[user_id]]).astype('float32')
    distances, indices = faiss_index.search(user_embedding, top_n + 1)

    matches = []
    all_users = fetch_all_users()

    for idx, distance in zip(indices[0], distances[0]):
        if idx == user_id:
            continue  # Skip self

        if idx < len(all_users):
            candidate = all_users[idx]
            matches.append({
                "user_id": candidate[0],
                "name": candidate[1],
                "city": candidate[2],
                "profile_text": candidate[4],
                "similarity_score": round(float(1 - distance), 2)
            })

    return matches

def recommend_filtered_users(user_id, selected_interests, top_n=10):
    try:
        user = fetch_user_by_id(user_id)
        if user is None:
            return f"UserID {user_id} not found."
    except Exception as e:
        return str(e)

    user_embedding = np.array([embeddings[user_id]]).astype('float32')

    distances, indices = faiss_index.search(user_embedding, top_n * 5)  # Search a bit wider

    recommended_users = []
    all_users = fetch_all_users()

    for idx in indices[0]:
        if idx == user_id:
            continue  # Skip self

        if idx < len(all_users):
            candidate = all_users[idx]
            candidate_interests = candidate[4].split()

        # Check intersection
        if any(interest in candidate_interests for interest in selected_interests):
            recommended_users.append({
                'user_id': candidate[0],
                'name': candidate[1],
                'city': candidate[2],
                'profile_text': candidate[4],
                'similarity_score': round(float(1 - distances[0][np.where(indices[0]==idx)[0][0]]), 2)
            })

        if len(recommended_users) >= top_n:
            break

    return recommended_users
