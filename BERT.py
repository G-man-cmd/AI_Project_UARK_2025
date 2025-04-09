import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

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

# Dataset Class
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

# Training Arguments
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train Model
trainer.train()

# Evaluate
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))