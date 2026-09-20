import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

# 1. Load and Prepare Data
df = pd.read_csv('data/train.csv')
# For binary classification, let's create a single 'toxic' label if any toxicity flag is 1
toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['is_toxic'] = df[toxicity_columns].max(axis=1)

X = df['comment_text'].fillna("unknown").values.tolist()
y = df['is_toxic'].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Baseline Model: TF-IDF + Logistic Regression
print("Training Baseline (TF-IDF + Logistic Regression)...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Using class_weight='balanced' handles the imbalance for the baseline
lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

baseline_preds = lr_model.predict(X_test_tfidf)
baseline_f1 = f1_score(y_test, baseline_preds, average='weighted')
print(f"Baseline Weighted F1-Score: {baseline_f1:.4f}")

# 3. Modern Architecture: DistilBERT fine-tuning
print("Preparing DistilBERT...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 4. Handling Class Imbalance with Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Custom Trainer to inject weights into the Loss Function
class ImbalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 5. Define Model and Metrics
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3, 
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = ImbalancedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("Training DistilBERT...")
trainer.train()

# 6. Save the final model for the web app
trainer.save_model("./toxic_comment_model")
tokenizer.save_pretrained("./toxic_comment_model")
print("Model saved to ./toxic_comment_model")