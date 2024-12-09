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
    prediction = model.predict(final_features)
    if prediction > .5:
        return render_template('index.html',
                               prediction_text='High chance of heart disease in 10-years'.format(
                                   prediction),
                               )
    else:
        return render_template('index.html',
                               prediction_text='Low chance of heart disease in 10-years'.format(
                                   prediction),
                               )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
