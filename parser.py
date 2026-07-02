import os
import json
import numpy as np
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=api_key)

CV_SCHEMA = {
    "type": "object",
    "properties": {
        "ad_soyad":   {"type": "string"},
        "iletisim": {
            "type": "object",
            "properties": {
                "e_posta":  {"type": "string"},
                "telefon":  {"type": "string"},
            },
        },
        "deneyim_yili": {"type": "number"},
        "yetenekler":   {"type": "array", "items": {"type": "string"}},
        "egitim":       {"type": "array", "items": {"type": "string"}},
        "ozet":         {"type": "string"},
    },
}


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)


def parse_cv_with_gemini(cv_text: str) -> dict:
    prompt = (
        "Aşağıdaki CV metnini analiz et ve yalnızca JSON formatında döndür.\n\n"
        f"{cv_text}"
    )
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CV_SCHEMA,
        ),
    )
    return json.loads(response.text)


def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(np.dot(v1, v2) / norm) * 100
