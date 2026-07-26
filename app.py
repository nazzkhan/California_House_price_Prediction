from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import sklearn

model_path = r"C:/Users/user/Hirarichal Clustering/california_info.joblib"
info = joblib.load(model_path)
model = info["model"]
columns = info["columns"]
ICONS = {
    "MedInc": "💰",
    "HouseAge": "🏚️",
    "AveRooms": "🛋️",
    "AveBedrms": "🛏️",
    "Population": "👥",
    "AveOccup": "👨‍👩‍👧",
    "Latitude": "📍",
    "Longitude": "🧭",
}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", columns=columns, icons=ICONS)

@app.route("/predict", methods=["GET"])
def predict():
    Input = []
    for i in columns:
        val = request.args.get(i, type=float)
        if val is None:
            return jsonify({"error": f"Missing or invalid value for '{i}'"}), 400
        Input.append(val)

    arr = np.array(Input)
    output = model.predict(arr.reshape(1, -1))

    return jsonify({"MedHouseVal": round(float(output[0]), 4)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)