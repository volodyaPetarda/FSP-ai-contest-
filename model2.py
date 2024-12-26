from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os


class TextClassifier:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = AutoModelForSequenceClassification.from_pretrained(dir_path)
        self.tokenizer = AutoTokenizer.from_pretrained(dir_path)
        self.labels = ['политика', 'туризм и путешествия', 'здоровье', 'наука и техника',
                       'развлечения', 'спорт']

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
        return self.labels[predicted_label]

    def predict_batch(self, texts):
        encodings = self.tokenizer(
            texts,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return [self.labels[prediction] for prediction in predictions]