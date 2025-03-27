from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
import re
import string

# Example preprocess_text function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = " ".join(text.split())
        # Add more preprocessing steps if needed (e.g., stemming, stopwords removal)
    return text


# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news_text']
    cleaned_text = preprocess_text(news_text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    result = "Real News" if prediction[0] == 1 else "Fake News"
    return render_template('index.html', prediction_text=f'This is {result}')

if __name__ == '__main__':
    app.run(debug=True)