from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ تحميل الموديل من المجلد الصحيح
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({
        "message": "✅ MLflow Heart Disease Prediction API is running successfully!",
        "usage": "Send POST request to /predict with full JSON feature names."
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ✅ تحويل البيانات إلى DataFrame
        df = pd.DataFrame([data])

        # ✅ إضافة العمود المفقود Unnamed: 0 إذا لم يكن موجودًا
        if "Unnamed: 0" not in df.columns:
            df["Unnamed: 0"] = 0

        # ✅ ترتيب الأعمدة لتطابق التدريب
        expected_features = [
            'Unnamed: 0', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
            'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex',
            'AgeCategory', 'PhysicalActivity', 'GenHealth', 'SleepTime',
            'Asthma', 'KidneyDisease', 'SkinCancer', 'Race_Asian',
            'Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White',
            'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
            'Diabetic_Yes (during pregnancy)'
        ]

        # ✅ ملء القيم المفقودة بـ 0 لتجنب NaN errors
        df = df.reindex(columns=expected_features, fill_value=0)

        # ✅ التنبؤ
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": int(prediction),
            "message": "✅ Prediction successful!"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # ✅ عند التشغيل المحلي
    app.run(host="0.0.0.0", port=5000)
