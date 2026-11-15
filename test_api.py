import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

client = genai.Client()

def ask_gemini(question: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction="Ты — молодежный рэпер который шарит за трэнды. Отвечай на русском языке.",
            temperature=0.2,
        ),
    )
    return response.text

if __name__ == "__main__":
    answer = ask_gemini("Зачитай рэпчину про самое святое что есть у пацана - бмв.")
    print(answer)