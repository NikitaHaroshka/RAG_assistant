import os
from dotenv import load_dotenv
from google import genai
import chromadb
import fitz

load_dotenv()
client = genai.Client()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="manalysis_textbook")

def extract_text_from_pdf(pdf_path: str):
    print(f"Чтение: {pdf_path}...")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += f"\n--- Страница {page_num + 1} ---\n" + page.get_text()
    return full_text

def chunk_text_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks

pdf_filename = "manalysis.pdf"

if not os.path.exists(pdf_filename):
    print(f"Ошибка: файл '{pdf_filename}' не найден в папке проекта!")
    print("Пожалуйста, положите PDF-файл учебника по матанализу в корень проекта и назовите его manalysis.pdf.")
else:
    raw_text = extract_text_from_pdf(pdf_filename)
    
    chunks = chunk_text_with_overlap(raw_text, chunk_size=1000, overlap=200)
    print(f"Учебник успешно разбит на {len(chunks)} смысловых чанков.")

    limit_chunks = 25
    print(f"Начинаем векторизацию первых {limit_chunks} чанков учебника через Gemini API...")

    for i in range(min(limit_chunks, len(chunks))):
        chunk = chunks[i]
        
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        vector = response.embeddings[0].values
        \
        collection.add(
            ids=[f"math_chunk_{i}"],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{"source": "calculus.pdf", "chunk_id": i}]
        )
        print(f"   - Чанк {i+1}/{limit_chunks} векторизован и сохранен.")

    print(f"\nГотово! Первые фрагменты учебника матанализа загружены в векторную базу ./chroma_db.")