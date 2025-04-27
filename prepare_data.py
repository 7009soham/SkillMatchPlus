# prepare_data.py

import pandas as pd
import numpy as np
import ast
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Paths
raw_data_path = 'SocialMediaUsersDataset.csv'  # Your original file
processed_data_path = 'backend/processed_dataset.csv'
embeddings_path = 'backend/embeddings.npy'
model_dir = 'backend/models'
model_path = os.path.join(model_dir, 'friendship_model.pkl')

# Step 1: Load and clean dataset
dataset = pd.read_csv(raw_data_path)

def clean_interests(row):
    try:
        interests = ast.literal_eval("[" + row + "]")
        interests = list(set([i.strip().strip("'") for i in interests]))
        return interests
    except:
        return []

dataset['Cleaned_Interests'] = dataset['Interests'].apply(clean_interests)
dataset['Profile_Text'] = dataset['Cleaned_Interests'].apply(lambda x: ' '.join(x))

# Save processed dataset
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
dataset.to_csv(processed_data_path, index=False)
print("✅ Saved:", processed_data_path)

# Step 2: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(dataset['Profile_Text'].tolist())

# Save embeddings
np.save(embeddings_path, embeddings)
print("✅ Saved:", embeddings_path)

# Step 3: Train simple logistic regression friendship model
def compute_similarity(user1_idx, user2_idx):
    inter1 = set(dataset.iloc[user1_idx]['Cleaned_Interests'])
    inter2 = set(dataset.iloc[user2_idx]['Cleaned_Interests'])
    common = len(inter1.intersection(inter2))
    total = len(inter1.union(inter2))
    return common / total if total != 0 else 0

pairs = []
similarities = []
labels = []

for _ in range(1000):
    i, j = random.sample(range(len(dataset)), 2)
    sim = compute_similarity(i, j)
    pairs.append((i, j))
    similarities.append(sim)
    labels.append(int(sim > 0.3))

X_train, X_test, y_train, y_test = train_test_split(similarities, labels, test_size=0.2, random_state=42)
X_train = [[x] for x in X_train]
X_test = [[x] for x in X_test]

friendship_model = LogisticRegression()
friendship_model.fit(X_train, y_train)

# Save model
os.makedirs(model_dir, exist_ok=True)
joblib.dump(friendship_model, model_path)
print("✅ Saved:", model_path)

print("\n🎯 All files created successfully!")
