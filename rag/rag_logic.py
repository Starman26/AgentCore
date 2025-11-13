import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from rag.db_access import retrieve_chat_summary, retrieve_student_info #see if you should set it kind of like a @tool

PERSIST_DIR = "robot_vector_db"
COLLECTION_NAME = "robot_problems"

def general_chat_db_use(chat_id : int):
    """create or update vectorStore for chat summary"""
    chat_docs, chat_docs_len = retrieve_chat_summary(chat_id)
    return create_or_update_vectorstore("chat_summary", chat_docs, chat_docs_len)
 
def general_student_db_use(name_or_email : str):
    """create or update vectorStore for student"""
    student_docs, student_docs_len = retrieve_student_info(name_or_email)
    return create_or_update_vectorstore("student_info", student_docs, student_docs_len)
    
def create_or_update_vectorstore(
    collection_name : str, 
    docs : list[Document], 
    docs_len: int 
):
    """Crea o actualiza el vector store basado en la DB."""  
    #add docs from certain table
    embedding = OpenAIEmbeddings()

    # Caso 1: si la base ya existe, cargarla y revisar tamaño
    if os.path.exists(PERSIST_DIR):
        print(" Vector database found, checking for updates...")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=PERSIST_DIR
        )
        
        # Verificar si hay más filas en CSV que documentos existentes
        current_count = len(vectorstore.get()["ids"])
        if current_count < docs_len:
            print(f" Updating database with {docs_len - current_count} new entries...")
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
            collection_name=collection_name,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()
        print("Vector database created successfully.")
    
    return vectorstore

