"""Flask API for churn inference - kept separate from Streamlit dashboard."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL_PATH = ROOT / "artifacts" / "models" / "sklearn_pipeline_mlflow.joblib"
model = None


def load_model():
    global model
    if MODEL_PATH.exists():
        import joblib
        model = joblib.load(MODEL_PATH)
    else:
        model = None


@app.route("/ping", methods=["GET"])
def ping():
    return "pong"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request"}), 400
        import pandas as pd
        X = pd.DataFrame([data])
        prob = float(model.predict_proba(X)[0, 1])
        pred = 1 if prob >= 0.35 else 0
        return jsonify({"prediction": pred, "probability": prob})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
