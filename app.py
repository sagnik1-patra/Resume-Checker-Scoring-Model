from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import fitz  # PyMuPDF
import os

app = Flask(__name__)

# Load model and tokenizer
model_dir = os.path.join(os.path.dirname(__file__), "resume_model")
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizerFast.from_pretrained(model_dir)
label2id = {'Data Science': 0, 'HR': 1, 'Advocate': 2, 'Arts': 3, 'Web Designing': 4,
            'Mechanical Engineer': 5, 'Sales': 6, 'Health and fitness': 7, 'Civil Engineer': 8,
            'Java Developer': 9, 'Business Analyst': 10, 'SAP Developer': 11, 'Automation Testing': 12,
            'Electrical Engineering': 13, 'Operations Manager': 14, 'Python Developer': 15,
            'DevOps Engineer': 16, 'Network Security Engineer': 17, 'PMO': 18, 'Database': 19,
            'Hadoop': 20, 'ETL Developer': 21, 'DotNet Developer': 22, 'Blockchain': 23, 'Testing': 24}
id2label = {v: k for k, v in label2id.items()}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def predict_resume(text):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    with torch.no_grad():
        logits = model(**encoding).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        confidence = probs[pred_idx]
    return id2label[pred_idx], confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["resume"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    text = extract_text_from_pdf(file_path)
    category, confidence = predict_resume(text)

    os.remove(file_path)  # Clean up uploaded file
    return jsonify({
        "category": category,
        "confidence": f"{confidence*100:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True)