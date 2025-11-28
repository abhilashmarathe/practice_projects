# app.py

import streamlit as st
import PyPDF2, docx2txt, pdfplumber, pytesseract
from PIL import Image
import re
import nltk
import spacy

# NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy NLP
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# ---------------------------
# Skills List
# ---------------------------
skills_list = [
    "python","java","c","c++","c#","javascript","html","css","react","angular","node",
    "django","flask","spring boot","php","laravel","ruby","rails","go","kotlin","swift",
    "typescript","mysql","postgresql","sql server","mongodb","oracle","firebase","git",
    "github","docker","kubernetes","aws","azure","gcp","linux","ci/cd","microservices",
    "devops","selenium","machine learning","deep learning","data science","nlp","computer vision",
    "tensorflow","pytorch","keras","pandas","numpy","matplotlib","seaborn","tableau","power bi",
    "excel","statistics","data analysis","big data","spark","hadoop","scikit-learn","jupyter notebook",
    "mlops","manual testing","automation testing","test cases","test planning","test strategy",
    "junit","postman","jira","regression testing","api testing","business development","market research",
    "sales","digital marketing","social media marketing","seo","sem","google analytics","campaign management",
    "crm","business strategy","negotiation","presentation","graphic design","ui design","ux design",
    "figma","adobe xd","photoshop","illustrator","premiere pro","after effects","video editing",
    "autocad","solidworks","catia","ansys","revit","staad pro","hvac","plc","quality control",
    "manufacturing","nursing","medical coding","clinical research","biotechnology","microbiology",
    "pharmacology","healthcare management","teaching","curriculum design","training","mentoring",
    "accounting","tally","sap","financial analysis","auditing","budgeting","quickbooks","payroll",
    "gst","communication","teamwork","leadership","problem solving","time management","adaptability",
    "decision making"
]

# ---------------------------
# Predefined Role-to-Skills Mapping
# ---------------------------
role_skill_map = {
    "data analyst": {
        "skills": ["python", "sql", "excel", "tableau", "power bi", "aws", "git", "data science", "numpy", "pandas"],
        "experience": 2
    },
    "software engineer": {
        "skills": ["python", "java", "c++", "git", "docker", "aws", "react", "angular"],
        "experience": 2
    },
    "machine learning engineer": {
        "skills": ["python", "tensorflow", "pytorch", "keras", "numpy", "pandas", "scikit-learn", "data science"],
        "experience": 3
    },
    # Add more roles as needed
}

# ---------------------------
# Helper Functions
# ---------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def extract_text(file):
    if file.name.endswith(".pdf"):
        text = ""
        try:
            # Try pdfplumber first
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + " "
            return text
        except:
            # Fallback to PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    elif file.type.startswith("image/"):
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    else:
        return ""

def extract_skills(text):
    return [skill for skill in skills_list if skill in text]

def extract_experience(text):
    patterns = [
        r'(\d+)\s*\+?\s*years?',
        r'(\d+)\s*yrs',
        r'(\d+)\s*year\s*(\d+)\s*months?',
        r'(\d+)\s*months?'
    ]
    experiences = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            experiences.extend(matches)
    return experiences

def calculate_experience(matches):
    total_years = 0
    for m in matches:
        if isinstance(m, tuple):
            years = int(m[0])
            months = int(m[1])
            total_years += years + months/12
        else:
            total_years += int(m)
    return round(total_years,2)

def extract_required_skills(job_text):
    job_text_lower = job_text.lower()
    # Check predefined roles first
    for role, info in role_skill_map.items():
        if role in job_text_lower:
            return info["skills"]
    # Fallback: extract skills from text
    return [skill for skill in skills_list if skill in job_text_lower]

def extract_required_experience(job_text):
    job_text_lower = job_text.lower()
    # Check predefined roles first
    for role, info in role_skill_map.items():
        if role in job_text_lower:
            return info["experience"]
    # Fallback: regex
    match = re.search(r'(\d+)\+?\s*years?', job_text_lower)
    return int(match.group(1)) if match else 0

def skill_match_score(candidate_skills, job_skills):
    if not job_skills:
        return 0
    matched_skills = set(candidate_skills).intersection(set(job_skills))
    return round(len(matched_skills)/len(job_skills)*100,2)

def calculate_overall_score(skill_score, exp_score, skill_weight=0.7, exp_weight=0.3):
    return round(skill_score*skill_weight + exp_score*exp_weight, 2)

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Resume Parser & Job Match", layout="wide")
st.title("💼 Resume Parser + Job Role Match Score")
st.write("Upload Resume (PDF, DOCX, or Image) and Paste Job Description")

resume_file = st.file_uploader("Upload Resume", type=["pdf","docx","png","jpg","jpeg"])
job_description = st.text_area("Paste Job Description Here")

if st.button("Calculate Match"):

    if not resume_file:
        st.warning("Please upload a resume file!")
    elif not job_description.strip():
        st.warning("Please paste job description!")
    else:
        # ------------------ Resume Processing ------------------
        raw_text = extract_text(resume_file)
        cleaned_text = clean_text(raw_text)
        lemmatized_text = lemmatize_text(cleaned_text)
        candidate_skills = extract_skills(lemmatized_text)
        experience_matches = extract_experience(lemmatized_text)
        candidate_exp = calculate_experience(experience_matches)

        # Add experience from date ranges
        total_years_from_dates = 0
        date_ranges = re.findall(r'(\d{4})\s*[–-]\s*(\d{4})', raw_text)
        for start, end in date_ranges:
            total_years_from_dates += int(end) - int(start)
        candidate_exp += total_years_from_dates

        # ------------------ Job Description Processing ------------------
        job_cleaned = clean_text(job_description)
        job_lemmatized = lemmatize_text(job_cleaned)
        job_skills = extract_required_skills(job_description)
        required_exp = extract_required_experience(job_description)

        # ------------------ Scoring ------------------
        skill_score = skill_match_score(candidate_skills, job_skills)
        exp_score = min(candidate_exp/required_exp,1)*100 if required_exp>0 else 0
        overall_score = calculate_overall_score(skill_score, exp_score)

        # Missing Skills
        missing_skills = list(set(job_skills) - set(candidate_skills))

        # ------------------ Display Results ------------------
        st.subheader("Candidate Skills")
        st.write(", ".join(candidate_skills))

        st.subheader("Required Skills")
        st.write(job_skills)

        st.subheader("Missing Skills")
        st.write(missing_skills)

        st.subheader("Experience")
        st.write(f"Candidate: {round(candidate_exp,2)} years | Required: {required_exp} years")

        st.subheader("Scores")
        st.write(f"Skill Match Score: {skill_score}%")
        st.write(f"Experience Score: {round(exp_score,2)}%")
        st.write(f"Overall Job Match Score: {overall_score}%")
