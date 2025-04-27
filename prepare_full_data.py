import pandas as pd
import numpy as np
import ast
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import random

# Paths
backend_path = 'backend'
dataset_path = os.path.join(backend_path, "processed_dataset.csv")
embeddings_path = os.path.join(backend_path, "embeddings.npy")
model_dir = os.path.join(backend_path, "models")
model_path = os.path.join(model_dir, "friendship_model.pkl")

# Step 1: Load Raw Dataset
raw_dataset = pd.read_csv('SocialMediaUsersDataset.csv')

# Step 2: Clean Interests
def clean_interests(row):
    try:
        interests = ast.literal_eval("[" + row + "]")
        interests = list(set([i.strip().strip("'") for i in interests]))
        return interests
    except:
        return []

raw_dataset['Cleaned_Interests'] = raw_dataset['Interests'].apply(clean_interests)
raw_dataset['Profile_Text'] = raw_dataset['Cleaned_Interests'].apply(lambda x: ' '.join(x))

# Step 3: Save processed dataset temporarily
os.makedirs(backend_path, exist_ok=True)
raw_dataset.to_csv(dataset_path, index=False)

# Step 4: Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(raw_dataset['Profile_Text'].tolist())
np.save(embeddings_path, embeddings)  

# Step 5: Create Clusters
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)
raw_dataset['Interest_Cluster'] = clusters

# Step 6: Save final processed dataset with clusters
raw_dataset.to_csv(dataset_path, index=False)

# Step 7: Train simple friendship model
# 1️⃣ First: Only compute similarity
def compute_similarity(user1_idx, user2_idx):
    try:
        inter1 = set(raw_dataset.iloc[user1_idx]['Cleaned_Interests'])
        inter2 = set(raw_dataset.iloc[user2_idx]['Cleaned_Interests'])
        common = len(inter1.intersection(inter2))
        total = len(inter1.union(inter2))
        similarity = common / total if total != 0 else 0
        return float(similarity)
    except Exception as e:
        print(f"Error comparing users {user1_idx} and {user2_idx}: {e}")
        return 0.0

# 2️⃣ Second: Predict based on similarity
def predict_friendship(user1_idx, user2_idx):
    similarity = compute_similarity(user1_idx, user2_idx)
    print(f"DEBUG: Similarity between {user1_idx} and {user2_idx} is {similarity:.2f}")

    if similarity >= 0.3:
        return "Strong Collaboration Likely"
    else:
        return "Weak Collaboration Likely"



pairs = []
similarities = []
labels = []

for _ in range(1000):
    i, j = random.sample(range(len(raw_dataset)), 2)
    sim = compute_similarity(i, j)
    pairs.append((i, j))
    similarities.append(sim)
    labels.append(int(sim > 0.3))

X_train, X_test, y_train, y_test = train_test_split(similarities, labels, test_size=0.2, random_state=42)
X_train = [[x] for x in X_train]
X_test = [[x] for x in X_test]

friendship_model = LogisticRegression()
friendship_model.fit(X_train, y_train)

os.makedirs(model_dir, exist_ok=True)
joblib.dump(friendship_model, model_path)

print("\n🎯 Successfully created:")
print(f"- {dataset_path}")
print(f"- {embeddings_path}")
print(f"- {model_path}")
