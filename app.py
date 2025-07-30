from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

#load the pretrained model
model = joblib.load("models/premium_model.pkl")
transformer = joblib.load("models/Transformer.pkl")
selector = joblib.load("models/Feature_selector.pkl")

#define a prediction function
def predict_premium(car_value,car_age,owner_age,vehicle_type,use_type,ownership_count,accident_history):
    #create a dataframe from the user input data
    input_df = pd.DataFrame([{
        "CarValue":float(car_value),
        "CarAge":float(car_age),
        "OwnerAge":int(owner_age),
        "VehicleType":vehicle_type,
        "UseType":use_type,
        "OwnershipCount":int(ownership_count),
        "AccidentHistory":int(accident_history)
    }])
    #transform the necessary features
    transformed = transformer.transform(input_df)
    selected = selector.transform(transformed)

    #make prediction and return the result
    prediction = model.predict(selected)
    return round(prediction[0],2)

#backend/frontend routes
@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        car_value=request.form["car_value"]
        car_age=request.form["car_age"]
        owner_age=request.form["owner_age"]
        vehicle_type=request.form["vehicle_type"]
        use_type=request.form["use_type"]
        ownership_count=request.form["ownership_count"]
        accident_history=request.form["accident_history"]
        accident_history = 1 if accident_history.lower() == "true" else 0

        #get premium prediction
        premium = predict_premium(car_value, car_age, owner_age, vehicle_type, use_type, ownership_count, accident_history)
        return render_template("index.html", prediction = premium)

    return render_template("index.html", prediction=None)
if __name__ == "__main__":
    app.run(debug=True)