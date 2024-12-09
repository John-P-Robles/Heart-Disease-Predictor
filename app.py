import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('templates/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_proba = model.predict_proba(final_features)
    prediction_probability = prediction_proba[0][1]  # Probability of the positive class
    
    threshold = 0.5  # You may need to adjust this
    
    if prediction_probability > threshold:
        risk_level = 'High'
    else:
        risk_level = 'Low'
    
    return render_template('index.html',
                           prediction_text=f'{risk_level} chance of heart disease in 10-years (Probability: {prediction_probability:.2f})')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
