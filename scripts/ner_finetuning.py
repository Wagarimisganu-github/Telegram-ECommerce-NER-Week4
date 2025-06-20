import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd

def load_conll_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n\n')
    
    sentences, labels = [], []
    for sentence in data:
        words, tags = [], []
        for line in sentence.split('\n'):
            word, tag = line.split()
            words.append(word)
            tags.append(tag)
        sentences.append(words)
        labels.append(tags)
    
    return sentences, labels

def prepare_dataset(sentences, labels):
    df = pd.DataFrame({'tokens': sentences, 'ner_tags': labels})
    dataset = Dataset.from_pandas(df)
    return dataset

def get_label_encodings():
    label_list = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-PRICE', 'I-PRICE']
    label2id = {label: id for id, label in enumerate(label_list)}
    id2label = {id: label for label, id in label2id.items()}
    return label_list, label2id, id2label

def load_model_and_tokenizer(model_name, num_labels, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return tokenizer, model

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def setup_trainer(model, tokenizer, tokenized_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    return trainer

def predict_ner(text, model, tokenizer, id2label):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    predicted_labels = [id2label[prediction.item()] for prediction in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return list(zip(tokens, predicted_labels))