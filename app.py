from flask import Flask, request, render_template
import pickle

# Load the trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function (reuse the one from your training code)
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

# Define routes
@app.route('/')
def index():
    return render_template('index.html')  # Load HTML form

@app.route('/predict', methods=['POST'])
def predict():
    # Get the news text from the form
    news = request.form['news']
    # Preprocess and transform the input text
    preprocessed_news = preprocess_text(news)
    text_tfidf = tfidf.transform([preprocessed_news])
    # Make prediction
    prediction = model.predict(text_tfidf)
    result = "Hurray!! It is Real" if prediction[0] == 1 else "Oops!! It is Fake"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
