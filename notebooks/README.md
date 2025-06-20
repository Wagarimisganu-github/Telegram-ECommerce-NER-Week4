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


## Notebooks Folder/ 

```
    `preprocessing_analysis.ipynb`                            : contains a notebook for or amharic data processing
    `ner_labelling.ipynb`                                     : contains a notebook to label amharic tokens in conll format
    `ner_finetuning.ipynb`                                    : contains a notebook to fine-tune a NER model 
    `model_comparison.ipynb`                                  : contains a notebook to compares different models NER
    `qenashcom_sinayelj_leyueqa_model_comparison.ipynb`       : contains a notebook to compares different models on combined conll data
    
```


## Setup Instructions
1. Clone the repository.
2. Set up the virtual environment.
3. Install dependencies using `pip install -r requirements.txt`


## Usage

### 1. Data Ingestion and Preprocessing

Run the data ingestion script to fetch and preprocess data from Telegram channels:

```
python scripts/telegram_scrapper.py

```

### 2. Data Labeling

Label a subset of the dataset in CoNLL format:

```
jupyter notebook notebooks/ner_labelling.ipynb
```

### 3. Fine-tuning NER Models

Fine-tune the NER model using the labeled dataset:

```
jupyter notebook notebooks/ner_finetuning.ipynb
```

### 4. Model Comparison and Selection

Compare different models and select the best-performing one:

```
jupyter notebook notebooks/model_comparison.ipynb
```

### 5. Model Interpretability

Analyze model predictions using SHAP and LIME:

```

```

## Dataset
The dataset used for training the custom tokenizer can be found [here]().
