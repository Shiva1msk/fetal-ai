from flask import Flask, request, render_template
import joblib  # For loading the model

app = Flask(__name__)

# Load the model using joblib (make sure the model file path is correct)
try:
    model = joblib.load('fetal_health.pkl')  # Update the path as needed
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to display the prediction form
@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")

# Route to handle form submission and perform prediction
@app.route("/output", methods=["POST"])
def output():
    if model is None:
        return render_template("error.html", message="Model not loaded. Please check the server.")

    try:
        # Extract and validate input from the form
        prolongued_decelerations = float(request.form.get('prolongued_decelerations', 0))
        abnormal_short_term_variability = float(request.form.get('abnormal_short_term_variability', 0))
        percentage_abnormal_long_term_variability = float(request.form.get('percentage_abnormal_long_term_variability', 0))
        histogram_variance = float(request.form.get('histogram_variance', 0))
        histogram_median = float(request.form.get('histogram_median', 0))
        mean_long_term_variability = float(request.form.get('mean_long_term_variability', 0))
        histogram_mode = float(request.form.get('histogram_mode', 0))
        accelerations = float(request.form.get('accelerations', 0))

        # Prepare data for prediction
        x = [[prolongued_decelerations, abnormal_short_term_variability, 
              percentage_abnormal_long_term_variability, histogram_variance, 
              histogram_median, mean_long_term_variability, histogram_mode, 
              accelerations]]

        # Perform prediction
        prediction = model.predict(x)
        labels = ['NORMAL', 'PATHOLOGICAL', 'SUSPECT']
        
        # Get result based on prediction
        result = labels[int(prediction[0])]
        print(f"Prediction result: {result}")

    except ValueError as ve:
        # Handle any ValueErrors
        return render_template('error.html', message=f"Invalid input or prediction: {ve}")
    except Exception as e:
        # Handle any other errors
        return render_template('error.html', message="An error occurred during prediction. Please try again.")

    # Render the output page with the result
    return render_template('output.html', output=result)

# Main entry point to run the app
if __name__ == "__main__":
    app.run()
    
# sample_input = {
#     'prolongued_decelerations': 0.002,  # between 0.000 and 0.005
#     'abnormal_short_term_variability': 50,  # between 12.0 and 87.0
#     'percentage_abnormal_long_term_variability': 45,  # between 0.0 and 91.0
#     'histogram_variance': 134,  # between 0.0 and 269.0
#     'histogram_median': 130,  # between 77.0 and 186.0
#     'mean_long_term_variability': 25,  # between 0.0 and 50.7
#     'histogram_mode': 120,  # between 60.0 and 187.0
#     'accelerations': 0.01  # between 0.000 and 0.019

# }
