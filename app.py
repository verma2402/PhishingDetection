from flask import Flask, render_template, request
import joblib
from feature_extraction2 import generate_data_set  # Make sure this function exists in your file

app = Flask(__name__)
model = joblib.load('pickle/model.pkl')


# Mapping model output to labels (matches your training labels: 1 & -1)
label_map = {1: 'Legitimate', -1: 'Phishing'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url', '')
    try:
        features = generate_data_set(url)
        pred = model.predict([features])[0]
        result = label_map.get(pred, 'Suspicious')
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('index.html', prediction_text=f"The URL is: {result}")

if __name__ == '__main__':
    app.run(debug=True)
