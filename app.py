from flask import Flask, request, jsonify
import joblib
import numpy as np
import json

# === Load model pipeline dan label map ===
model = joblib.load("svm_pca_pipeline.pkl")

with open("label_map.json", "r") as f:
    label_map = json.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "SVM + PCA Signal Classification API is running 🚀"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        features = data.get("features", None)

        if features is None:
            return jsonify({"error": "No features provided"}), 400

        features = np.array(features, dtype=np.float32)

        # Pastikan fitur 500
        if features.shape[0] != 500:
            return jsonify({"error": f"Expected 500 features, got {features.shape[0]}"}), 400

        # Reshape untuk model input (1, 500)
        features = features.reshape(1, -1)

        # Pipeline langsung transform + predict
        probs = model.predict_proba(features)[0]
        pred_class = int(np.argmax(probs))
        pred_label = label_map[str(pred_class)]

        return jsonify({
            "predicted_class": pred_class,
            "predicted_label": pred_label,
            "probabilities": probs.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
