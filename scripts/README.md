# Telegram-ECommerce-NER

## Overview

This project aims to develop a Named Entity Recognition (NER) system for EthioMart, a centralized e-commerce platform in Ethiopia. The system will extract key business entities such as product names, prices, and locations from Amharic text, images, and documents shared across multiple Telegram channels.


## Business Need

EthioMart's vision is to become the primary hub for all Telegram-based e-commerce activities in Ethiopia. By consolidating real-time data from multiple e-commerce Telegram channels, EthioMart aims to provide a seamless experience for customers to explore and interact with multiple vendors in one place.

## Key Objectives

1. Real-time data extraction from Telegram channels
2. Fine-tuning Large Language Models (LLMs) for Amharic Named Entity Recognition
3. Extraction of entities such as Product names, Prices, and Locations

## Project Structure

The project is divided into the following main tasks:

1. Data Ingestion and Preprocessing
2. Data Labeling in CoNLL Format
3. Fine-tuning NER Models
4. Model Comparison and Selection
5. Model Interpretability


## Folder Structure

- `notebooks/` : Jupyter notebooks for all the analysis.
- `scripts/`   : Python scripts for the notebook files .


## scripts Folder/ 

```
   `telegram_scrapper.py`                         : contains a script to fetch  data from Telegram channels
    `preprocessing.py`                            : contains a script for or amharic data processing
    `ner_labelling.py`                            : contains a script to label amharic tokens in conll format
    `ner_finetuning.py`                           : contains a script to fine-tune a NER model 
    `model_comparison.py`                         : contains a script to compares different models NER
    
    
```


## Setup Instructions
1. Clone the repository.
2. Set up the virtual environment.
3. Install dependencies using `pip install -r requirements.txt`


