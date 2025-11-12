import os
import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from .db_access import retrieve_robot_support #see if you should set it kind of like a @tool

PERSIST_DIR = "robot_vector_db"
COLLECTION_NAME = "robot_problems"


def create_or_update_vectorstore():
    """Crea o actualiza el vector store basado en el CSV."""
    print(" Checking vector database...")

    df = retrieve_robot_support()
    print(f"Datos obtenidos de la DB: {len(df)} registros")
    print(f"Columnas disponibles: {list(df.columns)}")
    if not df.empty:
        print(f"Primer registro: {df.iloc[0].to_dict()}")
        
    # Crear documentos
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "created_at": row["created_at"],
            "robot_type": row["robot_type"],
            "problem_title": row["problem_title"],
            "author": row["author"]
        }
        content = f"Problem: {row['problem_description']}\nSolution: {row['solution_steps']}"
        docs.append(Document(page_content=content, metadata=metadata))

    embedding = OpenAIEmbeddings()

    # Caso 1: si la base ya existe, cargarla y revisar tamaño
    if os.path.exists(PERSIST_DIR):
        print(" Vector database found, checking for updates...")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding,
            persist_directory=PERSIST_DIR
        )
        
        # Verificar si hay más filas en CSV que documentos existentes
        current_count = len(vectorstore.get()["ids"])
        if current_count < len(df):
            print(f" Updating database with {len(df) - current_count} new entries...")
            new_docs = docs[current_count:]  # solo los nuevos
            vectorstore.add_documents(new_docs)
            vectorstore.persist()
        else:
            print(" No updates needed.")
    else:
        print("Creating new vector database...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()
        print("Vector database created successfully.")
    
    return vectorstore