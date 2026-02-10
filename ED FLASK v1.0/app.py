from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Prepare tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Global variables for model
tfidf_final = None
rf_model_final = None
encoder_final = None

def nettoyer_texte(text):
    """Clean and process text"""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def regrouper_sentiments(emotion):
    """Group emotions into sentiment categories"""
    positives = ['positive', 'joy', 'excitement', 'contentment', 'gratitude', 'serenity', 'happy',
                 'hopeful', 'acceptance', 'enthusiasm', 'pride', 'elation', 'determination', 'surprise',
                 'playful', 'hope', 'happiness', 'proud', 'arousal', 'fulfillment', 'grateful', 'admiration',
                 'love', 'amusement', 'harmony', 'creativity', 'kind', 'enjoyment', 'adoration', 'resilience',
                 'coziness', 'contemplation', 'festivejoy', 'optimism', 'motivation', 'joyfulreunion',
                 'overjoyed', 'wonderment', 'appreciation', 'blessed', 'freedom', 'dazzle', 'playfuljoy',
                 'amazement', 'success', 'friendship', 'romance', 'celebration', 'positivity', 'kindness',
                 'triumph', 'renewed effort', 'vibrancy', 'relief']
    
    negatives = ['despair', 'loneliness', 'grief', 'awe', 'indifference', 'ambivalence', 'regret', 'hate',
                 'betrayal', 'frustration', 'frustrated', 'embarrassed', 'sad', 'bad', 'melancholy', 'bitterness',
                 'disgust', 'overwhelmed', 'negative', 'desolation', 'shame', 'dismissive', 'heartbreak',
                 'devastated', 'resentment', 'bitter', 'envious', 'fearful', 'helplessness', 'intimidation',
                 'anxiety', 'anger', 'fear', 'sadness', 'envy', 'disappointed', 'disappointment', 'mischievous',
                 'sorrow', 'loss', 'apprehensive', 'suffering', 'heartache', 'desperation', 'darkness', 'solitude',
                 'exhaustion', 'lostlove', 'solace', 'obstacle', 'miscalculation']
    
    if emotion in positives:
        return 'positive'
    elif emotion in negatives:
        return 'negative'
    else:
        return 'neutral'

def predire_sentiment(phrase):
    """Predict sentiment of a phrase"""
    global tfidf_final, rf_model_final, encoder_final
    
    if tfidf_final is None or rf_model_final is None or encoder_final is None:
        return 'neutral'
    
    phrase_clean = nettoyer_texte(phrase)
    phrase_sparse = tfidf_final.transform([phrase_clean])
    phrase_vectorized = phrase_sparse.toarray()  # type: ignore
    
    prediction_id = rf_model_final.predict(phrase_vectorized)[0]
    sentiment = encoder_final.inverse_transform([prediction_id])[0]
    
    return sentiment

def train_model(df_clean):
    """Train the sentiment analysis model"""
    global tfidf_final, rf_model_final, encoder_final
    
    # Vectorization
    tfidf_final = TfidfVectorizer(max_features=5000)
    X_final_sparse = tfidf_final.fit_transform(df_clean['cleaned_Text'])
    X_final = X_final_sparse.toarray()  # type: ignore
    
    # Encoding
    encoder_final = LabelEncoder()
    y_final = encoder_final.fit_transform(df_clean['sentiment_category'])
    
    # Split
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42
    )
    
    # Training
    rf_model_final = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_final.fit(X_train_final, y_train_final)

def process_csv_file(filepath):
    """Process CSV file and return sentiment counts"""
    try:
        # Load the file
        df = pd.read_csv(filepath, on_bad_lines='skip')
        
        # Check if required columns exist
        if 'Text' not in df.columns or 'Sentiment' not in df.columns:
            return None, "CSV must contain 'Text' and 'Sentiment' columns"
        
        # Select useful columns
        df_clean = df[['Text', 'Sentiment']]
        
        # Check for missing values and remove empty rows
        df_clean = df_clean.dropna(subset=['Text', 'Sentiment'])
        
        if len(df_clean) == 0:
            return None, "No valid data found in the CSV file"
        
        # Clean text
        df_clean['cleaned_Text'] = df_clean['Text'].apply(nettoyer_texte)
        
        # Clean sentiment labels
        df_clean['Sentiment'] = df_clean['Sentiment'].astype(str).str.strip().str.lower()
        
        # Apply grouping
        df_clean['sentiment_category'] = df_clean['Sentiment'].apply(regrouper_sentiments)
        
        # Train model
        train_model(df_clean)
        
        # Count sentiments for all texts
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for phrase in df_clean['Text']:
            resultat = predire_sentiment(phrase)
            sentiment_counts[resultat] += 1
        
        return sentiment_counts, None
    
    except pd.errors.ParserError as e:
        return None, f"CSV parsing error: {str(e)}"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '' or file.filename is None:
        return jsonify({'error': 'No file selected'}), 400
    
    if not isinstance(file.filename, str) or not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400
    
    try:
        filename = secure_filename(file.filename) if file.filename else 'upload.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        sentiment_counts, error = process_csv_file(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if error:
            return jsonify({'error': error}), 400
        
        if sentiment_counts is None:
            return jsonify({'error': 'Failed to process file'}), 400
        
        return jsonify({
            'positive': sentiment_counts.get('positive', 0),
            'neutral': sentiment_counts.get('neutral', 0),
            'negative': sentiment_counts.get('negative', 0)
        }), 200
    
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
