import os
import pandas as pd
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


PERSIST_DIR = "robot_vector_db"
CSV_PATH = "./robot_problems.csv"
COLLECTION_NAME = "robot_problems"

def create_or_update_vectorstore():
    """Crea o actualiza el vector store basado en el CSV."""
    print("🔍 Checking vector database...")

    # Cargar CSV
    df = pd.read_csv(CSV_PATH)

    # Crear documentos
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "created_at": row["created_at"],
            "robot_type": row["robot_type"],
            "problem_title": row["problem_title"],
            "author": row["author"]
        }
        content = f"Problem: {row['problem_description']}\nSolution: {row['solution_description']}"
        docs.append(Document(page_content=content, metadata=metadata))

    embedding = OpenAIEmbeddings()

    # Caso 1: si la base ya existe, cargarla y revisar tamaño
    if os.path.exists(PERSIST_DIR):
        print("✅ Vector database found, checking for updates...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=PERSIST_DIR
        )
        
        # Verificar si hay más filas en CSV que documentos existentes
        current_count = len(vectorstore.get()["ids"])
        if current_count < len(df):
            print(f"⚙️ Updating database with {len(df) - current_count} new entries...")
            new_docs = docs[current_count:]  # solo los nuevos
            vectorstore.add_documents(new_docs)
            vectorstore.persist()
        else:
            print("✅ No updates needed.")
    else:
        print("🆕 Creating new vector database...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()
        print("✅ Vector database created successfully.")
    
    return vectorstore