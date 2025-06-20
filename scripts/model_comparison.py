import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from seqeval.metrics import classification_report

def load_conll_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n\n')
    
    sentences, labels = [], []
    for sentence in data:
        words, tags = [], []
        for line in sentence.split('\n'):
            if line.strip():
                try:
                    word, tag = line.split()
                    words.append(word)
                    tags.append(tag)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line}")
        if words and tags:
            sentences.append(words)
            labels.append(tags)
    
    return sentences, labels

def prepare_dataset(sentences, labels):
    df = pd.DataFrame({'tokens': sentences, 'ner_tags': labels})
    dataset = Dataset.from_pandas(df)
    return dataset

def get_label_encodings():
    label_list = ['O', 'B-Product', 'I-Product', 'B-LOC', 'I-LOC', 'B-Price', 'I-Price']
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

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=128):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        is_split_into_words=True
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(label[word_idx], -100))
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    return trainer


def finetune_and_evaluate(model_name, train_dataset, eval_dataset, label_list, label2id, id2label):
    tokenizer, model = load_model_and_tokenizer(model_name, len(label_list), id2label, label2id)
    
    def tokenize_and_align_dataset(dataset):
        # Convert pandas DataFrame to Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(dataset)
        
        tokenized_dataset = hf_dataset.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, label2id),
            batched=True,
            remove_columns=hf_dataset.column_names
        )
        return tokenized_dataset
    
    train_tokenized = tokenize_and_align_dataset(train_dataset)
    eval_tokenized = tokenize_and_align_dataset(eval_dataset)
    
    output_dir = f"./results_{model_name.split('/')[-1]}"
    trainer = setup_trainer(model, tokenizer, train_tokenized, eval_tokenized, output_dir)
    
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    eval_results = trainer.evaluate()
    
    # Compute additional metrics
    predictions = trainer.predict(eval_tokenized)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    
    report = classification_report(true_labels, true_predictions)
    
    return {
        "model_name": model_name,
        "eval_loss": eval_results["eval_loss"],
        "training_time": training_time,
        "classification_report": report,
        "model": model,
        "tokenizer": tokenizer
    }

def predict_ner(text, model, tokenizer, id2label):
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    predicted_labels = [id2label[prediction.item()] for prediction in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return list(zip(tokens, predicted_labels))