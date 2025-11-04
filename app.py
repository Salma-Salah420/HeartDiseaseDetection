from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)

# Load the model from MLflow directory
model = mlflow.pyfunc.load_model("model")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ MLflow model serving is running."})

@app.route("/invocations", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if "dataframe_split" not in data:
            return jsonify({"error": "Invalid input format. Expected 'dataframe_split'."}), 400

        df = pd.DataFrame(**data["dataframe_split"])
        preds = model.predict(df)
        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
