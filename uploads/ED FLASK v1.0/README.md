# Sentiment Analysis Flask App

A web application for analyzing sentiment in CSV files using machine learning.

## Features
- Upload CSV files with Text and Sentiment columns
- Automatic sentiment analysis and classification
- Display results for positive, neutral, and negative sentiments
- Beautiful, responsive UI

## Requirements
- Python 3.7+
- Flask
- pandas
- scikit-learn
- nltk

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (automatically done on first run, but you can pre-download):
```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Running the Application

1. Navigate to the project directory:
```bash
cd "path\to\ED FLASK"
```

2. Run the Flask app:
```bash
python app.py
```

3. Open your browser and go to:
```
http://localhost:5000
```

## CSV File Format

Your CSV file should contain at least these two columns:
- **Text**: The text content to analyze
- **Sentiment**: The sentiment label (e.g., happy, sad, positive, negative, etc.)

Example:
```csv
Text,Sentiment
"I love this product",positive
"This is terrible",negative
"It's okay",neutral
```

## Sentiment Categories

Sentiments are automatically grouped into three categories:
- **Positive**: joy, happiness, love, excitement, etc.
- **Negative**: sadness, anger, fear, disappointment, etc.
- **Neutral**: Everything else

## Project Structure

```
ED FLASK/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # HTML template
└── static/
    ├── style.css         # CSS styling
    └── script.js         # JavaScript functionality
```

## Notes

- Maximum file size: 16MB
- The application trains a Random Forest classifier on your data
- Processing time depends on file size
