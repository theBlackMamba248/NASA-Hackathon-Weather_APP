from flask import Flask, request, jsonify
import logging
from utils import get_recommendation, get_weather_data

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)




# --- New Recommendation API ---
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    location = data.get('location')
    date = data.get('date')
    preferences = data.get('preferences')

    if not location or not date or not preferences:
        return jsonify({'error': 'Missing required fields'}), 400

    weather = get_weather_data(date, location)
    row = {
        "preferred_activities": preferences,
        "weather": weather
    }
    recommendation = get_recommendation(row)

    response = {
        "location": location,
        "date": date,
        "weather": weather,
        "recommendation": recommendation
    }
    return jsonify(response)



if __name__ == '__main__':
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        logging.error(f"Failed to start the Flask app: {str(e)}")
