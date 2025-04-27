from flask import Blueprint, jsonify
from services.matching_service import get_top_matches
from services.community_service import get_same_community_users
from services.friendship_service import predict_friendship

match_bp = Blueprint('match', __name__)

@match_bp.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    matches = get_top_matches(user_id)
    return jsonify(matches), 200



@match_bp.route('/community/<int:user_id>', methods=['GET'])
def community(user_id):
    community_users = get_same_community_users(user_id)
    return jsonify(community_users), 200

@match_bp.route('/predict_friendship/<int:user1_id>/<int:user2_id>', methods=['GET'])
def predict_friendship_route(user1_id, user2_id):
    result = predict_friendship(user1_id, user2_id)
    print(f"FRIENDSHIP PREDICTION DEBUG: User {user1_id} vs User {user2_id}: {result}")
    return jsonify({"prediction": result}), 200
