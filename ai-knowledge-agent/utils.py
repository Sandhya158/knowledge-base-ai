import os
import numpy as np
import faiss
import fitz
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_pdf(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text += t
    if text.strip():
        return text
    return ocr_pdf(path)

def ocr_pdf(path):
    pdf = fitz.open(path)
    text = ""
    for page in pdf:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
    return text

def load_documents(folder="data"):
    docs = []
    for f in os.listdir(folder):
        if f.lower().endswith(".pdf"):
            full = os.path.join(folder, f)
            d = extract_text_pdf(full)
            docs.append({"page_content": d, "name": f})
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = []
    for d in docs:
        chunks.extend(splitter.split_text(d["page_content"]))
    return chunks

def embed_list(texts):
    vectors = []
    for t in texts:
        if not t.strip():
            continue
        r = genai.embed_content(
            model="models/text-embedding-004",
            content=t
        )
        vectors.append(r["embedding"])
    return np.array(vectors).astype("float32")

def create_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index

def search(query, index, chunks, k=4):
    r = genai.embed_content(
        model="models/text-embedding-004",
        content=query
    )
    q = np.array(r["embedding"]).astype("float32")
    D, I = index.search(np.array([q]), k)
    results = [chunks[i] for i in I[0]]
    score = float(np.mean(D))
    return results, score

def generate_answer(context, query, tone, history):
    model = genai.GenerativeModel("gemini-2.5-flash")

    hist = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        hist.append({"role": role, "parts": [h["content"]]})

    prompt = f"{tone}\nUse ONLY this context:\n{context}\n\nUser: {query}"

    chat = model.start_chat(history=hist)
    r = chat.send_message(prompt)
    return r.text
