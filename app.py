import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
import nltk
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download necessary NLTK data
nltk.download('punkt')

# Load stopwords from CSV
stopwords_df = pd.read_csv('hindi_stopwords.csv')
stopwords_set = set(stopwords_df['stopword'].tolist())  # Adjust 'stopword' if needed

# Function to preprocess text
def preprocess_text(text):
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove links starting with 'https'
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # Remove English words (assuming only Hindi text is desired)
    text = re.sub(r'[a-zA-Z]+', '', text)
    # Remove numeric values
    text = re.sub(r'\d+', '', text)
    # Tokenization using NLTK
    tokens = word_tokenize(text, language='english')

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords_set]
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Join tokens back to text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Define reverse mapping dictionary for labels
reverse_sentiment_mapping = {
    1: 'defamation',
    2: 'fake',
    3: 'hate',
    4: 'non-hostile',
    5: 'offensive',
    6: 'multiple'  # Label for comments with multiple sentiments
}

# Load the preprocessed data
df = pd.read_csv("Preprocessed_HindiComments.csv")

# Handle missing values
df['Preprocessed_Post'] = df['Preprocessed_Post'].fillna("") # Split the data into features and labels
X = df['Preprocessed_Post']
y = df['Label']

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
# Fit and transform the text data
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# HTML Form for input and output
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Hate Speech Detection</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
            color: #333;
        }}
        .container {{
            width: 50%;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            margin-top: 50px;
        }}
        h1 {{
            text-align: center;
            color: #4CAF50;
        }}
        label {{
            font-weight: bold;
        }}
        textarea {{
            width: 100%;
            height: 100px;
            padding: 10px;
            margin-top: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        input[type="submit"] {{
            display: block;
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }}
        input[type="submit"]:hover {{
            background-color: #45a049;
        }}
        .result {{
            margin-top: 20px;
            padding: 10px;
            background-color: #e7f3e7;
            border-left: 5px solid #4CAF50;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hate Speech Detection in Hindi</h1>
        <form action="/predict" method="post">
            <label for="comment">Enter a speech to test:</label><br>
            <textarea id="comment" name="comment"></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
        {result_section}
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def read_form():
    return html_form.format(result_section="")

@app.post("/predict", response_class=HTMLResponse)
def predict(comment: str = Form(...)):
    # Preprocess the new comment
    preprocessed_new_comment = preprocess_text(comment)

    # Transform the preprocessed comment using the TfidfVectorizer
    X_new_comment = tfidf_vectorizer.transform([preprocessed_new_comment])

    # Predict the label of the new comment
    predicted_label_encoded = lr_model.predict(X_new_comment)
    predicted_label_name = reverse_sentiment_mapping.get(predicted_label_encoded[0], 'unknown')

    result_section = f"""
    <div class="result">
        <h2>Predicted Label:</h2>
        <p>{predicted_label_name}</p>
    </div>
    """

    return html_form.format(result_section=result_section)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)


