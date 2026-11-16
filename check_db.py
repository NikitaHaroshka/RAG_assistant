import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="calculus_textbook")
all_data = collection.get(include=['documents', 'metadatas'])

# Открываем файл для записи (перезаписывает при каждом запуске)
with open("output.txt", "w", encoding="utf-8") as f:
    total_docs = len(all_data['ids'])
    f.write(f"📊 Всего документов в базе: {collection.count()}\n")


    f.write("\n🔍 Примеры сохраненных чанков:\n")
    for i in range(total_docs):
        doc_id = all_data['ids'][i]
        doc_text = all_data['documents'][i]
        metadata = metadata = all_data['metadatas'][i]

        f.write(f"\n--- ID: {doc_id} ---\n")
        f.write(f"Метаданные: {metadata}\n")
        f.write(f"Текст (первые 250 символов):\n{doc_text[:250]}...\n")

print("✅ Готово! Результат сохранён в output.txt")