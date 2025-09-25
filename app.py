
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load model
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Home page with form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction from API (POST JSON)
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({"prediction": int(prediction)})

# Prediction from HTML form
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        return render_template("index.html", prediction_text=f"Predicted Class: {prediction}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)