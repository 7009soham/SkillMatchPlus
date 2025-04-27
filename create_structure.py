import os

# Root directory
root_dir = r"D:\SKILLMATCHPLUS"

# Folder structure
folders = [
    "backend/models",
    "backend/services",
    "backend/controllers",
    "backend/utils",
    "backend/config",
    "backend/database",
    "frontend/public",
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/services",
    "postman_collections"
]

# Important files to create
files = [
    "backend/app.py",
    "backend/requirements.txt",
    "frontend/src/App.js",
    "frontend/src/index.js",
    "README.md",
    ".env"
]

# Create folders
for folder in folders:
    path = os.path.join(root_dir, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created folder: {path}")

# Create files
for file in files:
    path = os.path.join(root_dir, file)
    with open(path, 'w') as f:
        f.write("")  # Create an empty file
    print(f"Created file: {path}")

print("\n✅ All folders and files created successfully!")
