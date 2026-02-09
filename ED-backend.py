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
from sklearn.metrics import classification_report, accuracy_score


# 1. Load the file
df = pd.read_csv('input-file.csv', on_bad_lines='skip')

# 2. Select useful columns
df_clean = df[['Text', 'Sentiment']]

# 3. Check for missing values
df_clean.isnull().sum()

# 4. Remove empty rows
df_clean = df_clean.dropna(subset=['Text', 'Sentiment'])


# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Prepare tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Create cleaning function
def nettoyer_texte(text):
    # A. Convert to string and lowercase
    text = str(text).lower()
    
    # B. Remove non-letters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # C. Tokenize
    words = text.split()
    
    # D. Remove stopwords and apply stemming
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    # E. Rejoin
    return " ".join(cleaned_words)

# Apply cleaning
df_clean['cleaned_Text'] = df_clean['Text'].apply(nettoyer_texte)


# Clean sentiment labels
df_clean['Sentiment'] = df_clean['Sentiment'].astype(str).str.strip().str.lower()

def regrouper_sentiments(emotion):
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

# Apply grouping
df_clean['sentiment_category'] = df_clean['Sentiment'].apply(regrouper_sentiments)


# Vectorization
tfidf_final = TfidfVectorizer(max_features=5000)
X_final_sparse = tfidf_final.fit_transform(df_clean['cleaned_Text'])
X_final = np.asarray(X_final_sparse.toarray())  # type: ignore

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

# Prediction
y_pred_final = rf_model_final.predict(X_test_final)

# Define prediction function
def predire_sentiment(phrase):
    # Clean the phrase
    phrase_clean = nettoyer_texte(phrase)
    
    # Vectorize
    phrase_sparse = tfidf_final.transform([phrase_clean])
    phrase_vectorized = np.asarray(phrase_sparse.toarray())  # type: ignore
    
    # Predict
    prediction_id = rf_model_final.predict(phrase_vectorized)[0]
    
    # Get sentiment name
    sentiment = encoder_final.inverse_transform([prediction_id])[0]
    
    return sentiment

# Count sentiments for all texts in the dataset
sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

for phrase in df_clean['Text']:
    resultat = predire_sentiment(phrase)
    sentiment_counts[resultat] += 1

# Display results in the requested format
print("output:")
for sentiment in ['positive', 'neutral', 'negative']:
    count = sentiment_counts[sentiment]
    print(f"{sentiment:<12}{count}")
