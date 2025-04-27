import pandas as pd
import sqlite3

# Load your CSV
import os

# Auto-detect correct path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go one folder back to /backend
dataset_path = os.path.join(base_path, "processed_dataset.csv")

df = pd.read_csv(dataset_path)


# Connect to SQLite (creates new file if not exists)
conn = sqlite3.connect('skillmatch.db')
cursor = conn.cursor()

# Create users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        UserID INTEGER,
        Name TEXT,
        City TEXT,
        DOB TEXT,
        Profile_Text TEXT
    )
''')

# Insert data
for _, row in df.iterrows():
    cursor.execute('''
        INSERT INTO users (UserID, Name, City, DOB, Profile_Text)
        VALUES (?, ?, ?, ?, ?)
    ''', (row['UserID'], row['Name'], row['City'], row['DOB'], row['Profile_Text']))

# Commit and close
conn.commit()
conn.close()

print("✅ Database created and data migrated successfully!")
