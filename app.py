from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from preprocessing import preprocess_input

app = Flask(__name__)

# Load models and scaler
lr_model = joblib.load('models/lr_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')  # You need to save this during training

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'age': int(request.form['age']),
                'job': request.form['job'],
                'marital': request.form['marital'],
                'education': request.form['education'],
                'default': request.form['default'],
                'balance': float(request.form['balance']),
                'housing': request.form['housing'],
                'loan': request.form['loan'],
                'contact': request.form['contact'],
                'day': int(request.form['day']),
                'month': request.form['month'],
                'duration': int(request.form['duration']),
                'campaign': int(request.form['campaign']),
                'pdays': int(request.form['pdays']),
                'previous': int(request.form['previous']),
                'poutcome': request.form['poutcome']
            }

            # Preprocess the input
            processed_data = preprocess_input(input_data)

            # Make predictions
            lr_pred = lr_model.predict(processed_data)[0]
            svm_pred = svm_model.predict(processed_data)[0]

            # Convert predictions to human-readable format
            lr_result = "Yes (will subscribe)" if lr_pred == 1 else "No (will not subscribe)"
            svm_result = "Yes (will subscribe)" if svm_pred == 1 else "No (will not subscribe)"

            return render_template('index.html',
                                   lr_prediction=lr_result,
                                   svm_prediction=svm_result,
                                   show_results=True)

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html', show_results=False)

if __name__ == '__main__':
    app.run(debug=True)