#  Resume Selector

A Machine Learning powered tool that **classifies resumes into job roles** using NLP and BERT. It also provides a **web app** for recruiters to upload resumes (PDF) and get the predicted role with a confidence score and keyword analysis.

---

##  Features

 **Resume Classification**: Classifies resumes into 25 job roles.  
 **Keyword Analysis**: Highlights important keywords missing from resumes.  
 **Web App**: Upload PDF resumes and see predictions instantly.  
 **API**: REST API to integrate with other platforms.  
 **Easy Deployment**: Flask app ready for production.  

---

##  Demo

 _Add screenshots here once you run the app_  

1. Upload Resume PDF  
2. Get predicted role and confidence score  
3. See keyword suggestions for improvement  

---

##  Technologies Used

-  Python 3.11
-  HuggingFace Transformers (BERT)
-  PyTorch
-  SpaCy
-  Flask (for API + Web)
-  PyMuPDF (fitz) for PDF parsing
-  HTML/CSS + JavaScript

---

##  Project Structure

Resume Selector/
│
├── train_resume_model.py # Train BERT on resume dataset
├── predict_resume.py # Predict role from a resume text
├── app.py # Flask web server
├── requirements.txt # Python dependencies
├── resume_model/ # Saved BERT model & tokenizer
├── templates/
│ └── index.html # Frontend webpage
├── static/ # (Optional CSS/JS)
├── uploads/ # Temporary uploaded PDFs
└── README.md # This file

yaml
Copy
Edit

---

##  Setup Instructions

###  Clone Repository
```bash
git clone https://github.com/<your-username>/resume-selector.git
cd resume-selector
 Create Virtual Environment
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate    # On Windows
 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
python -m spacy download en_core_web_sm
 Train Model (Optional)
If you want to retrain:

bash
Copy
Edit
python train_resume_model.py
