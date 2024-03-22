# Import Flask from flask
from flask import Flask, render_template, request
import joblib
import sklearn

# Create the instance of a class
my_application = Flask(__name__)

# Create the route and bind them with respective function
@my_application.route("/")
def home():
    return render_template("home.html")

@my_application.route("/prediction", methods=["GET", "POST"])
def prediction_fun():
    if request.method == "POST":
        review = request.form.get("review")
        # Check if the review is empty
        if not review.strip():
            return render_template("home.html", error="Please enter a review.")
        
        model = joblib.load("Models/logistic_regression.pkl")
        prediction = model.predict([review])
        return render_template("output.html", prediction=prediction)
    else:
        # If the method is GET, redirect the user back to the home page
        return redirect("/")

# Run the application
if (__name__) == "__main__":
    my_application.run(debug = True, host="0.0.0.0")

