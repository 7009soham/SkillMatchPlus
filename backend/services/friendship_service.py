from database.db_connection import dataset, friendship_model
from datetime import datetime

def calculate_age(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception as e:
        print(f"Date parsing error for DOB '{dob_str}': {e}")
        return 0

def predict_friendship(user1_id, user2_id):
    if user1_id >= len(dataset) or user2_id >= len(dataset):
        return "Invalid users"

    # Parse interests
    interests1 = eval(dataset.iloc[user1_id]['Cleaned_Interests'])
    interests2 = eval(dataset.iloc[user2_id]['Cleaned_Interests'])

    inter1 = set(interests1)
    inter2 = set(interests2)

    common = len(inter1.intersection(inter2))
    total = len(inter1.union(inter2))
    jaccard_similarity = common / total if total != 0 else 0

    # 💥 Calculate age difference from DOB
    dob1 = dataset.iloc[user1_id]['DOB']
    dob2 = dataset.iloc[user2_id]['DOB']
    age1 = calculate_age(dob1)
    age2 = calculate_age(dob2)
    age_difference = abs(age1 - age2)

    # Same country
    country1 = dataset.iloc[user1_id]['Country']
    country2 = dataset.iloc[user2_id]['Country']
    same_country = 1 if country1 == country2 else 0

    # Gender match
    gender1 = dataset.iloc[user1_id]['Gender']
    gender2 = dataset.iloc[user2_id]['Gender']
    gender_match = 1 if gender1 == gender2 else 0

    # 📦 Make full input
    X_input = [[jaccard_similarity, age_difference, same_country, gender_match]]

    # 📈 Predict
    prediction = friendship_model.predict(X_input)

    return "Strong Collaboration Likely" if prediction[0] == 1 else "Weak Collaboration Likely"
