import os
import re
from dotenv import load_dotenv

from openai import OpenAI

# ✅ Updated LangChain imports (no deprecation)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------------
# 🔥 LLM SUMMARIZER
# -------------------------------
def llm_summarize(text, severity, risk):
    prompt = f"""
You are a medical assistant.

Patient condition:
- COPD Severity: {severity}
- Risk Score: {risk}

Using the following medical guideline text, generate:
- Clear
- Short (3–4 sentences)
- Patient-friendly advice

Guideline Text:
{text}

Output only clean advice. No headings, no copyright text.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# -------------------------------
# 📄 LOAD PDF
# -------------------------------
def load_medical_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


# -------------------------------
# ✂️ SPLIT DOCUMENTS
# -------------------------------
def split_docs(documents):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


# -------------------------------
# 🧼 CLEAN TEXT (STRONG CLEANER)
# -------------------------------
def clean_text(text):
    text = text.replace("\n", " ")

    # Remove ALL CAPS headings
    text = re.sub(r'\b[A-Z\s]{5,}\b', '', text)

    # Remove copyright junk
    text = re.sub(r'COPYRIGHT.*?DISTRIBUTE', '', text)

    # Remove numbers-only tokens
    text = re.sub(r'\b\d+\b', '', text)

    # Fix broken words
    text = text.replace("Pr evention", "Prevention")

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# -------------------------------
# 🧠 EXTRACT MEANINGFUL SENTENCES
# -------------------------------
def extract_useful_sentences(text):
    sentences = text.split(".")
    
    priority_keywords = [
        "oxygen", "bronchodilator", "rehabilitation",
        "inhaler", "corticosteroid"
    ]

    useful = []

    for s in sentences:
        s = s.strip().lower()

        for k in priority_keywords:
            if k in s:
                useful.append(s.capitalize())
                break

    return useful[:3]

# -------------------------------
# 📊 CREATE VECTOR DB
# -------------------------------
def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(docs, embeddings)


# -------------------------------
# 🔍 MAIN QUERY FUNCTION
# -------------------------------
def query_db(db, query, severity, risk):
    results = db.similarity_search(query, k=2)
    combined = " ".join([doc.page_content for doc in results])

    try:
        print("DEBUG → Using LLM...")
        return llm_summarize(combined, severity, risk)

    except Exception:
        print("⚠️ LLM failed → using clean fallback")

        cleaned = clean_text(combined)
        sentences = extract_useful_sentences(cleaned)

        advice = ". ".join(sentences)

        return f"""
Patient has {severity} COPD with risk score {risk}.

Recommended actions:
{advice}
"""