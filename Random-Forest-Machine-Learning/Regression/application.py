import pickle
import pandas as pd
from flask import Flask, render_template, request

application = Flask(__name__)
app = application

preprocessor = pickle.load(open("Models/preprocessor.pkl", "rb"))
model = pickle.load(open("Models/randomsearchcv.pkl", "rb"))

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":

        model_name = request.form.get("model")
        seller_type = request.form.get("seller_type")
        fuel_type = request.form.get("fuel_type")
        transmission_type = request.form.get("transmission_type")

        vehicle_age = float(request.form.get("vehicle_age"))
        km_driven = float(request.form.get("km_driven"))
        mileage = float(request.form.get("mileage"))
        engine = float(request.form.get("engine"))
        max_power = float(request.form.get("max_power"))
        seats = float(request.form.get("seats"))

        input_df = pd.DataFrame({
            "model": [model_name],
            "vehicle_age": [vehicle_age],
            "km_driven": [km_driven],
            "seller_type": [seller_type],
            "fuel_type": [fuel_type],
            "transmission_type": [transmission_type],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "seats": [seats]
        })

        transformed_data = preprocessor.transform(input_df)
        prediction = model.predict(transformed_data)

        return render_template("home.html", results=f"₹ {round(prediction[0], 2)}")

    return render_template("home.html")


if __name__ == "__main__":
    app.run()
