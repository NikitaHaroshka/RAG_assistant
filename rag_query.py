import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

load_dotenv()
client = genai.Client()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="calculus_textbook")

def ask_rag(user_query: str):
    print(f"\n Вопрос пользователя: «{user_query}»")
    
    query_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=user_query
    )
    query_vector = query_response.embeddings[0].values
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )
    
    retrieved_chunks = results['documents'][0]
    
    print(f"Найдено релевантных чанков из учебника: {len(retrieved_chunks)}")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"   - Чанк {i+1} (начало): {chunk[:120]}...")

    context = "\n\n".join(retrieved_chunks)
    
    system_prompt = (
        "Ты — строгий и квалифицированный ассистент преподавателя математического анализа. "
        "Используй ТОЛЬКО предоставленный контекст из учебника, чтобы ответить на вопрос студента. "
        "Если в контексте нет ответа или информации недостаточно, честно скажи об этом. "
        "Не придумывай формулы, теоремы или доказательства от себя."
    )
    
    user_prompt = f"Контекст из учебника:\n{context}\n\nВопрос студента: {user_query}"

    print("Отправляем контекст и вопрос в Gemini (gemini-3.8-flash)...")
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    )
    
    return response.text

if __name__ == "__main__":
    question = "Расскажи что такое предел функции и приведи мне критерий монотонности."
    answer = ask_rag(question)
    
    print("\n======================")
    print(answer)
    