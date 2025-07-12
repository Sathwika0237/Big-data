import pickle
import flask
from flask import Flask, render_template, request


# Load and use later
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Example usage
sample_tweet = ["I'm really enjoying this!"]
features = vectorizer.transform(sample_tweet)
prediction = model.predict(features)

print("Sentiment:", prediction[0])

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        vect = vectorizer.transform([tweet])
        prediction = model.predict(vect)[0]
        return render_template('index.html', prediction=prediction,
                               tweet=tweet)

if __name__ == '__main__':
    app.run(debug=True)

