from flask import Flask, render_template, request
import pickle
import numpy as np
model = pickle.load(open("rfc_model.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        Pregnancies = request.form["Pregnancies"]
        Glucose = request.form["Glucose"]
        BloodPressure = request.form["BloodPressure"]
        SkinThickness = request.form["SkinThickness"]
        Insulin = request.form["Insulin"]
        BMI = request.form["BMI"]
        DiabetesPedigreeFunction = request.form["DiabetesPedigreeFunction"]
        Age = request.form["Age"]

        arr = np.array([[Pregnancies, Glucose, BloodPressure,
                       SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        output = model.predict(arr)
        return render_template('result.html', prediction=output)


if __name__ == "__main__":
    app.run(debug=True)
