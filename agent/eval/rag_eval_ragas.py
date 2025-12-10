"""
rag_eval_ragas.py

Evaluación del RAG de FrEDIE usando ragas.

- Usa tus tools reales: retrieve_context y search_in_db
- Construye un dataset de (question, contexts, answer, ground_truth)
- Evalúa con métricas ragas (answer_relevancy, faithfulness, context_precision, context_recall)

Ejecuta:
    python rag_eval_ragas.py
"""

import os
import json
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 👇 importa tus tools tal cual están en tu proyecto
from Settings.tools import (
    retrieve_context,
    search_in_db,
)

# ==========================
# Carga de entorno
# ==========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY para ragas eval")


# ==========================
# CASOS DE PRUEBA (edita aquí)
# ==========================
"""
Define aquí los casos que quieres evaluar.

Campos sugeridos por caso:
- id: identificador corto del caso
- description: contexto humano (para ti)
- question: la pregunta que hará el estudiante
- student: email o nombre registrado en tu tabla students
- chat_id: id numérico del chat asociado (el mismo que usas para general_chat_db_use)
- project_id: proyecto industrial/lab (para search_in_db)
- team_id: equipo (para filtrar agent_tables)
- ground_truth: respuesta esperada / referencia
- use_student_ctx: bool, si quieres que use retrieve_context
- use_db_ctx: bool, si quieres que use search_in_db
"""

TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": "abb_sensores_practica_1",
        "description": "Sensores en la celda ABB para la práctica 1.",
        "question": (
            "Explica los sensores principales que se usan en la celda de trabajo "
            "del ABB IRB 1100 (internos y externos) y sus funciones."
        ),
        "student": "alumna@tec.mx",
        "chat_id": 1,  # ajusta al chat real si lo usas en general_chat_db_use
        "project_id": "TU_PROJECT_ID_O_NONE",
        "team_id": "TU_TEAM_ID_O_NONE",
        "ground_truth": (
            "Debe mencionar sensores internos (encoders, resolvers) para posición/velocidad, "
            "y sensores externos como de proximidad, cortinas de seguridad, botones de emergencia "
            "y, si aplica, cámaras/visión para la celda ABB."
        ),
        "use_student_ctx": True,
        "use_db_ctx": True,
    },
    # Añade más casos aquí...
]


# ==========================
# Helpers para obtener contexto RAG real
# ==========================

def get_rag_context_for_case(case: Dict[str, Any]) -> List[str]:
    """
    Usa tus tools reales para construir la lista de contexts que verá ragas.
    - retrieve_context: vectorstore de estudiante + chat
    - search_in_db: tablas internas (agent_tables)
    """
    question = case["question"]
    contexts: List[str] = []

    # --- 1) Contexto por estudiante + chat (retrieve_context) ---
    if case.get("use_student_ctx") and case.get("student") and case.get("chat_id") is not None:
        try:
            ctx = retrieve_context.invoke(
                {
                    "name_or_email": case["student"],
                    "chat_id": case["chat_id"],
                    "query": question,
                }
            )
            if isinstance(ctx, str) and ctx.strip() and ctx.strip() != "RAG_EMPTY":
                contexts.append(ctx)
        except Exception as e:
            print(f"[get_rag_context_for_case] error retrieve_context en {case['id']}: {e}")

    # --- 2) Contexto desde tablas de proyecto/equipo (search_in_db) ---
    if case.get("use_db_ctx"):
        try:
            db_ctx = search_in_db.invoke(
                {
                    "project_id": case.get("project_id"),
                    "team_id": case.get("team_id"),
                    "query": question,
                    "max_tables": 3,
                    "k_per_table": 4,
                    "only_tables": None,
                }
            )
            if isinstance(db_ctx, str) and db_ctx.strip() and not db_ctx.startswith("DB_CONTEXT::RAG_EMPTY"):
                contexts.append(db_ctx)
        except Exception as e:
            print(f"[get_rag_context_for_case] error search_in_db en {case['id']}: {e}")

    if not contexts:
        contexts = ["SIN_CONTEXTO_RAG"]  # para que ragas no truene

    return contexts


def generate_answer_from_context(question: str, contexts: List[str]) -> str:
    """
    Usa un LLM (el mismo modelo que FrEDIE, gpt-4o-mini) para generar
    una respuesta usando SOLO el contexto RAG.
    Esto simula cómo contestaría tu agente idealmente.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    # Unimos todos los contexts en un solo bloque grande.
    # Si quieres, aquí puedes aplicar un truncado o limpieza.
    joined_context = "\n\n".join(contexts)

    system_prompt = (
        "Eres un asistente técnico de FrEDIE. Debes contestar usando SOLO el contexto "
        "proporcionado. Si algo no está en el contexto, dilo explícitamente y no inventes.\n\n"
        "Responde en español, con explicaciones claras y técnicas pero entendibles."
    )

    user_prompt = (
        f"Contexto RAG:\n{joined_context}\n\n"
        f"Pregunta del estudiante:\n{question}\n\n"
        "Responde de forma completa pero concisa:"
    )

    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return (resp.content or "").strip()


# ==========================
# Construir Dataset para ragas
# ==========================

def build_ragas_dataset(cases: List[Dict[str, Any]]) -> Dataset:
    """
    Construye un HuggingFace Dataset con:
    - question
    - answer (generada con el LLM usando tu RAG)
    - contexts (lista de strings)
    - ground_truth (respuesta esperada)
    """
    rows: List[Dict[str, Any]] = []

    for case in cases:
        print(f"\n=== Caso: {case['id']} ===")
        question = case["question"]
        ground_truth = case["ground_truth"]

        # 1) Obtener contexto real del RAG
        contexts = get_rag_context_for_case(case)
        print(f"  > Contextos obtenidos: {len(contexts)}")

        # 2) Generar respuesta con ese contexto
        answer = generate_answer_from_context(question, contexts)
        print(f"  > Respuesta generada (preview): {answer[:120]}...")

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "case_id": case["id"],
            }
        )

    dataset = Dataset.from_list(rows)
    return dataset


# ==========================
# Main: correr evaluación
# ==========================

def main():
    print("Construyendo dataset para ragas...")
    dataset = build_ragas_dataset(TEST_CASES)

    print("\nDataset listo. Filas:", len(dataset))

    # LLM y embeddings que usará ragas internamente
    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nEjecutando ragas.evaluate() ...\n")

    result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=eval_llm,
        embeddings=embeddings,
    )

    # result suele ser un EvaluationResult con métodos útiles
    print("\n===== MÉTRICAS GLOBALES =====")
    try:
        # Algunas versiones exponen .overall o similar
        print(result)
    except Exception:
        # Fallback si se comporta como dict
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"{k}: {v}")
        else:
            print("Resultado ragas:", result)

    # Detalle por ejemplo
    print("\n===== RESULTADOS POR CASO =====")
    try:
        df = result.to_pandas()
        # Mostrar bonito
        for idx, row in df.iterrows():
            cid = row.get("case_id", f"row_{idx}")
            print(f"\n--- Caso {cid} ---")
            for metric_name in ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]:
                if metric_name in row:
                    print(f"{metric_name}: {row[metric_name]:.3f}")
    except Exception as e:
        print("No se pudo convertir a DataFrame:", e)


if __name__ == "__main__":
    main()
