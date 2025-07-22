import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import spacy

#  Paths
data_path = r"C:\Users\sagni\Downloads\Resume Selector\UpdatedResumeDataSet.csv"
model_save_path = r"C:\Users\sagni\Downloads\Resume Selector\resume_model"

#  Load dataset
df = pd.read_csv(data_path, encoding='utf-8')
df = df[['Category', 'Resume']]
df.dropna(inplace=True)
print(f"Dataset shape: {df.shape}")

#  Encode labels
labels = df['Category'].unique()
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
df['label'] = df['Category'].map(label2id)

#  Text preprocessing
nlp = spacy.load('en_core_web_sm')
def preprocess(text):
    doc = nlp(str(text).lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

df['clean_resume'] = df['Resume'].apply(preprocess)

#  Dataset class
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts.iloc[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encodings.items()}, torch.tensor(self.labels.iloc[idx])

#  Tokenizer & Model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#  Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_resume'], df['label'], test_size=0.2, random_state=42)
train_dataset = ResumeDataset(X_train, y_train, tokenizer)
test_dataset = ResumeDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

#  Training
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 2

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, labels_batch = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels_batch = labels_batch.to(device)

        outputs = model(**inputs, labels=labels_batch)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

#  Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels_batch = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(labels_batch.numpy())

print("\n Classification Report:\n", classification_report(y_true, y_pred, target_names=labels))

#  Save model & tokenizer
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f" Model and tokenizer saved to '{model_save_path}'")