
import joblib
import pandas as pd
from flask import Flask, request, jsonify

learn_api = Flask("ExtraaLearn")

model = joblib.load("learn_model.joblib")

@learn_api.get("/")
def home():
    return "Welcome to the Lead Prediction System"

@learn_api.post("/v1/predict")
def predict_sales():
    data = request.get_json()

    sample = {
        "age": data["age"],
        "current_occupation": data["current_occupation"],
        "first_interaction": data["first_interaction"],
        "profile_completed": data["profile_completed"],
        "website_visits": data["website_visits"],
        "time_spent_on_website": data["time_spent_on_website"],
        "page_views_per_visit": data["page_views_per_visit"],
        "last_activity": data["last_activity"],
        "print_media_type1": data["print_media_type1"],
        "print_media_type2": data["print_media_type2"],
        "digital_media": data["digital_media"],
        "educational_channels": data["educational_channels"],
        "referral": data["referral"]
    }

    input_data = pd.DataFrame([sample])

    prediction = model.predict(input_data).tolist()[0]

    return jsonify({"Lead": int(prediction)})

if __name__ == "__main__":
    learn_api.run(debug=True)
