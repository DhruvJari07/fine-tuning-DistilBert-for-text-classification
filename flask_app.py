from flask import Flask, request, render_template
from predict import PredictPipeline

app = Flask(__name__)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get user input from the form
    user_input = request.form["user_input"]

    # Make prediction
    predictor = PredictPipeline()
    prediction = predictor.predict(user_input)

    # Render prediction result
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
