from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
lr_model = joblib.load("lr_model.pkl")
rf_model = joblib.load("rf_model.pkl")   # make sure this file exists
scaler = joblib.load("scaler.pkl")

feature_names = [
    "Temp", "Humidity", "Cloud Cover", "ANNUAL",
    "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec",
    "avgjune", "sub"
]


@app.route("/", methods=["GET", "POST"])
def home():
    probability = None
    warning = None
    selected_model = "lr"

    if request.method == "POST":
        selected_model = request.form["model"]

        values = [
            float(request.form["Temp"]),
            float(request.form["Humidity"]),
            float(request.form["CloudCover"]),
            float(request.form["ANNUAL"]),
            float(request.form["JanFeb"]),
            float(request.form["MarMay"]),
            float(request.form["JunSep"]),
            float(request.form["OctDec"]),
            float(request.form["avgjune"]),
            float(request.form["sub"])
        ]

        input_df = pd.DataFrame([values], columns=feature_names)

        if selected_model == "lr":
            input_scaled = scaler.transform(input_df)
            probability = lr_model.predict_proba(input_scaled)[0][1]
        else:
            probability = rf_model.predict_proba(input_df)[0][1]

        # Warning logic
        if probability >= 0.8:
            warning = "🚨 SEVERE FLOOD WARNING"
        elif probability >= 0.6:
            warning = "⚠️ MODERATE FLOOD RISK"
        elif probability >= 0.4:
            warning = "🟡 LOW FLOOD RISK"
        else:
            warning = "🟢 NO FLOOD RISK"

    return render_template(
        "index.html",
        probability=probability,
        warning=warning,
        selected_model=selected_model
    )
if __name__ == "__main__":
    app.run(debug=True)
