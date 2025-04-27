from database.db_connection import dataset

def get_same_community_users(user_id):
    if user_id >= len(dataset):
        return []
    
    user_cluster = dataset.iloc[user_id]['Interest_Cluster']
    community_members = dataset[dataset['Interest_Cluster'] == user_cluster]
    
    users = []
    for idx, row in community_members.iterrows():
        users.append({
            "user_id": int(row['UserID']),
            "name": row['Name']
        })
    
    return users
