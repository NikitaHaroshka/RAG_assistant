import os
import time
from dotenv import load_dotenv
from google import genai
import chromadb
import pymupdf


load_dotenv()
client = genai.Client()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="calculus_textbook")

def extract_text_from_pdf(pdf_path: str):
    print(f"Читаем PDF-файл: {pdf_path}...")
    doc = pymupdf.open(pdf_path)
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
else:
    raw_text = extract_text_from_pdf(pdf_filename)
    chunks = chunk_text_with_overlap(raw_text, chunk_size=1000, overlap=200)
    print(f"Учебник успешно разбит на {len(chunks)} смысловых чанков.")

    total_chunks = len(chunks)
    print(f"Начинаем векторизацию чанков через Gemini API...")

    for i in range(total_chunks):
        chunk = chunks[i]
        
        try:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=chunk
            )
            vector = response.embeddings[0].values
            
            collection.upsert(
                ids=[f"math_chunk_{i}"],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[{"source": "manalysis.pdf", "chunk_id": i}]
            )
            if (i + 1) % 10 ==0 or i == 0:
                print(f"   Прогресс: {i + 1} / {total_chunks}.")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Ошибка на чанке {i+1}: {e}")
            break

    print(f"\nГотово! Процесс загрузки завершен.")