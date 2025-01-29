from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import pandas as pd

# 1. Carregar a base de dados
data_path = 'path_to_your_dataset.csv'  # Altere para o caminho do dataset
data = pd.read_csv(data_path)

# Supondo que temos colunas 'text' e 'label'
texts = data['text'].tolist()
labels = data['label'].tolist()

# 2. Divisão entre treino, validação e teste
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

# 3. Tokenização
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Criando datasets do Hugging Face
dataset_train = Dataset.from_dict({"text": train_texts, "label": train_labels})
dataset_val = Dataset.from_dict({"text": val_texts, "label": val_labels})
dataset_test = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Aplicando a tokenização
dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)
dataset_test = dataset_test.map(tokenize_function, batched=True)

# Removendo a coluna de texto, mantendo apenas tokens
dataset_train = dataset_train.remove_columns(["text"])
dataset_val = dataset_val.remove_columns(["text"])
dataset_test = dataset_test.remove_columns(["text"])

# 4. Treinar o modelo BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(labels)))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    return {
        'accuracy': acc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    compute_metrics=compute_metrics
)

trainer.train()

# 5. Avaliação nos dados de teste
predictions = trainer.predict(dataset_test)
logits = predictions.predictions
predicted_labels = torch.argmax(torch.tensor(logits), dim=-1).tolist()

# Métricas finais
acc = accuracy_score(test_labels, predicted_labels)
f1_micro = f1_score(test_labels, predicted_labels, average='micro')
f1_macro = f1_score(test_labels, predicted_labels, average='macro')
conf_matrix = confusion_matrix(test_labels, predicted_labels)

print("Accuracy:", acc)
print("F1 Micro:", f1_micro)
print("F1 Macro:", f1_macro)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(test_labels, predicted_labels))
