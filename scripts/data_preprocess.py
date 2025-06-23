# scripts/data_preprocess.py
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import emoji
import json
import os
import sys

def clean_amharic_text(text):
    """
    Clean Amharic text by removing URLs, emojis, unwanted characters,
    and normalizing whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Keep Amharic Unicode range (1200-137F) and basic punctuation, letters, numbers
    text = re.sub(r'[^\w\s\u1200-\u137F.,!?]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def preprocess_data(raw_data_path):
    """
    Load raw JSON data, clean text, tokenize, and save to processed CSV.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")

    print(f"Loading data from: {raw_data_path}")
    with open(raw_data_path, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    df = pd.DataFrame(messages)
    
    # Clean text column
    df["cleaned_text"] = df["text"].apply(clean_amharic_text)
    
    # Tokenize cleaned text
    df["tokens"] = df["cleaned_text"].apply(word_tokenize)
    
    # Filter out empty or whitespace-only messages
    df = df[df["cleaned_text"].str.len() > 0]
    
    # Prepare output folder and path
    processed_dir = os.path.join(os.path.dirname(raw_data_path).replace("raw", "processed"))
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, os.path.basename(raw_data_path).replace(".json", ".csv"))
    
    # Save processed data
    df.to_csv(processed_path, index=False, encoding="utf-8")
    print(f"Processed data saved to: {processed_path}")
    
    return df

if __name__ == "__main__":
    # Use CLI arg if given, else default example path
    if len(sys.argv) > 1:
        raw_data_path = sys.argv[1]
    else:
        raw_data_path = "data/raw/telegram_messages_20250621_052911.json"
        print(f"No input file path given. Using default: {raw_data_path}")

    preprocess_data(raw_data_path)
