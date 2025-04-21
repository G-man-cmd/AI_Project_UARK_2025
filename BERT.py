import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

import torch.nn.functional as F
from collections import defaultdict

# Load and Clean Data
df = pd.read_csv("BooksDataSet.csv")
df = df.dropna(subset=["book_name", "summary", "genre"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

df["text"] = (df["book_name"] + " " + df["summary"]).apply(clean_text)

# Encode Labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["genre"])
num_labels = len(label_encoder.classes_)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Tokenize Text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=256)

class BookDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = BookDataset(train_encodings, y_train.tolist())
test_dataset = BookDataset(test_encodings, y_test.tolist())

# Load BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

def get_token_importance(text, model, tokenizer, label_id):
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer.encode_plus(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    baseline_logits = model(**inputs).logits[0]
    baseline_score = F.softmax(baseline_logits, dim=0)[label_id].item()
    
    importances = []
    for i in range(1, len(tokens) + 1):  # skip [CLS]
        masked = inputs["input_ids"].clone()
        masked[0, i] = tokenizer.mask_token_id
        logits = model(input_ids=masked, attention_mask=inputs["attention_mask"]).logits[0]
        masked_score = F.softmax(logits, dim=0)[label_id].item()
        importances.append((tokens[i - 1], baseline_score - masked_score))
    return sorted(importances, key=lambda x: -abs(x[1]))[:10]

genre_examples = defaultdict(str)
for genre in label_encoder.classes_:
    example = df[df["genre"] == genre].iloc[0]
    genre_examples[genre] = example["text"]

print("\nTop 10 Important Words by Genre (Token Masking):")
for genre in label_encoder.classes_:
    text = genre_examples[genre]
    label_id = label_encoder.transform([genre])[0]
    try:
        top_words = get_token_importance(text, model, tokenizer, label_id)
        print(f"\nBook Genre: {genre}")
        for word, score in top_words:
            print(f"{word:15} {score:.4f}")
    except Exception as e:
        print(f"Could not compute importance for {genre}: {e}")
