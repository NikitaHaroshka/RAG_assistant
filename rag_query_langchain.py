from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

embandings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function= embandings,
    collection_name="calculus_textbook"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.1
)

system_instruction = """Ты — учитель математики.

Отвечай на вопрос, используя ТОЛЬКО предоставленный контекст.

Правила:
1. Не выдумывай теоремы и формулы. 
2. Если ответа нет в контексте, честно скажи:
   "В предоставленных конспектах я не нашёл ответа."
3. Когда решаешь задачу, объясняй ход решения последовательно.
4. Формулы сохраняй в LaTeX.
5. Если в контексте есть доказательство подходящей теоремы,
   используй его как основу объяснения.
6. Не утверждай, что в конспекте сказано то, чего там нет.
7. Не добавляй никаких источников в ответ.
8. Отвечай только на русском языке.

Контекст:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    ("user", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

user_query = "Что такое интеграл?"
print(f"Вопрос: {user_query}\n")

response = rag_chain.invoke(user_query)

print(response)