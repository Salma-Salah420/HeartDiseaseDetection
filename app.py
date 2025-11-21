from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# ✅ تحميل الموديل من المجلد الصحيح
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({
        "message": "✅ Heart Disease Prediction API is running!",
        "usage": "Send POST request to /predict with input JSON features."
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # تحويل البيانات إلى DataFrame
        df = pd.DataFrame([data])

        # إضافة العمود المفقود إذا غير موجود
        if "Unnamed: 0" not in df.columns:
            df["Unnamed: 0"] = 0

        # ترتيب الأعمدة بنفس ترتيب التدريب
        expected_features = [
            'Unnamed: 0', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
            'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex',
            'AgeCategory', 'PhysicalActivity', 'GenHealth', 'SleepTime',
            'Asthma', 'KidneyDisease', 'SkinCancer', 'Race_Asian',
            'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White',
            'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
            'Diabetic_Yes (during pregnancy)'
        ]

        # ملء القيم الناقصة بـ 0
        df = df.reindex(columns=expected_features, fill_value=0)

        # التنبؤ
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": int(prediction),
            "message": "Prediction successful!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
