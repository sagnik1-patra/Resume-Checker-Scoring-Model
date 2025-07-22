import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import fitz  # PyMuPDF
import spacy

#  Load model & tokenizer
model_dir = r"C:\Users\sagni\Downloads\Resume Selector\resume_model"
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#  Role-specific keywords
role_keywords = {
    'Data Science': ['python', 'machine learning', 'pandas', 'tensorflow'],
    'Python Developer': ['python', 'flask', 'django', 'api'],
    'Java Developer': ['java', 'spring', 'hibernate'],
}

#  Preprocess function
nlp = spacy.load('en_core_web_sm')
def preprocess(text):
    doc = nlp(str(text).lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

#  Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

#  Predict and analyze
def predict_resume_from_pdf(pdf_path, threshold=0.85):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(" No text found in the PDF!")
        return

    print(" Extracted Resume Text (first 500 chars):\n", text[:500], "...\n")

    clean_text = preprocess(text)
    encoding = tokenizer(clean_text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = torch.argmax(torch.tensor(probs)).item()
        confidence = probs[pred_idx]
        predicted_category = list(role_keywords.keys())[pred_idx]

    print(f" Predicted Category: {predicted_category}")
    print(f" Confidence Score: {confidence * 100:.2f}%")

    #  Keyword Analysis
    resume_words = set(clean_text.split())
    keywords = set(role_keywords.get(predicted_category, []))
    present_keywords = resume_words & keywords
    missing_keywords = keywords - resume_words

    print(f" Found keywords: {', '.join(present_keywords) if present_keywords else 'None'}")
    if missing_keywords:
        print(f" Missing important keywords: {', '.join(missing_keywords)}")
    else:
        print(" All key skills present!")

#  Example Usage
pdf_resume_path = r"C:\Users\sagni\Downloads\Resume NextWave\Resume.pdf"
predict_resume_from_pdf(pdf_resume_path)