from flask import Flask, request, jsonify
import joblib
import numpy as np
import json

# === Load model, scaler, dan label map ===
svm_model = joblib.load("svm_signal_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

app = Flask(__name__)

# === Route utama ===
@app.route("/")
def home():
    return jsonify({"message": "SVM Signal Classification API is running 🚀"})

# === Endpoint prediksi ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get("features", None)

        if features is None:
            return jsonify({"error": "No features provided"}), 400

        # Pastikan panjang fitur = 500
        features = np.array(features, dtype=np.float32)
        if features.shape[0] != 500:
            return jsonify({"error": f"Expected 500 features, got {features.shape[0]}"}), 400

        # Scale
        features_scaled = scaler.transform([features])

        # Prediksi
        probs = svm_model.predict_proba(features_scaled)[0]
        pred_class = int(np.argmax(probs))
        pred_label = label_map[str(pred_class)]

        return jsonify({
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "probabilities": probs.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Run server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)