from flask import Blueprint, jsonify

test_bp = Blueprint('test_bp', __name__)

@test_bp.route('/', methods=['GET'])
def test_route():
    return jsonify({"message": "SkillMatch+ Backend is running successfully!"}), 200
