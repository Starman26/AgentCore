# rag_eval.py
# Evaluación simple de RAG usando LLM como juez.
# Colócalo junto a tu código del agente y ejecútalo con:
#   python rag_eval.py

import os
import json
import statistics
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Importa TUS tools reales
from Settings.tools import (
    retrieve_context,
    search_in_db,
)

load_dotenv()

# ============================
# 1) Configuración del juez
# ============================

judge_llm = ChatOpenAI(
    model="gpt-4o-mini",  # o el que uses como juez
    temperature=0,
)

JUDGE_PROMPT = """
Eres un evaluador de un sistema RAG (Retrieval-Augmented Generation).

Tu tarea es CALIFICAR solo la CALIDAD del CONTEXTO recuperado, NO la respuesta final del asistente.

Te doy:
- PREGUNTA_DEL_USUARIO
- CONTEXTO_RECUPERADO (texto que el sistema encontró en bases de datos / documentos)
- RESPUESTA_ESPERADA (breve descripción de lo que se debería poder responder con buen contexto)

Responde con un JSON **estricto** de la forma:
{{
  "score": <entero de 1 a 5>,
  "justificacion": "<explicación corta>"
}}

Criterio:
- 1: Contexto casi no relacionado, inútil para responder.
- 2: Algo relacionado pero muy incompleto o superficial.
- 3: Relevante pero faltan detalles importantes.
- 4: Muy relevante, con casi todo lo necesario.
- 5: Altamente relevante y suficiente para responder muy bien.

PREGUNTA_DEL_USUARIO:
\"\"\"{question}\"\"\"


CONTEXTO_RECUPERADO:
\"\"\"{context}\"\"\"


RESPUESTA_ESPERADA:
\"\"\"{expected}\"\"\"


Recuerda: SOLO devuelve el JSON, sin texto extra.
"""


# ====================================
# 2) Casos de prueba RAG (rellénalos)
# ====================================

# Puedes mezclar dos modos:
# - mode="student": usa retrieve_context (perfil + historial de chat)
# - mode="db": usa search_in_db (agent_tables / project_tasks / etc.)

TEST_CASES: List[Dict[str, Any]] = [
    # Ejemplo 1: RAG de prácticas (project_tasks, task_steps, etc.)
    {
        "id": "practica_sensores_robotica",
        "mode": "db",
        "question": "¿Cuáles son las tareas o prácticas de este proyecto de Sensores en robótica?",
        "expected_answer": "Debe encontrar en project_tasks las prácticas relacionadas al proyecto de Sensores en robótica y listar al menos los títulos de las tareas.",
        "project_id": "10c0cb99-b5a4-4f50-9a16-3f11e9611bd3",  # <-- cambia por tu project_id
        "team_id": None,
        "max_tables": 2,
        "k_per_table": 4,
        # "only_tables": ["project_tasks"],  # opcional
    },

    # Ejemplo 2: RAG por estudiante (perfil + historial)
    {
        "id": "perfil_leo_andrade",
        "mode": "student",
        "question": "¿Qué habilidades y metas tiene Leonardo Andrade?",
        "expected_answer": "Debe recuperar del perfil de Leonardo Andrade que sus skills incluyen Python y Solidworks, y que su meta es mejorar su aprendizaje en ML.",
        "student_email": "A01174639@tec.mx",  # ajusta al correo real
        "chat_id": 1,  # cualquier entero; tu tool internamente lo ignora o lo usa
    },

    # Agrega aquí más casos de prueba...
]


# ============================
# 3) Funciones de evaluación
# ============================

def get_rag_context(case: Dict[str, Any]) -> str:
    """
    Llama al RAG real según el modo indicado en el caso de prueba.
    """
    mode = case["mode"]

    # ---- Caso 1: RAG por estudiante (retrieve_context) ----
    if mode == "student":
        email = case["student_email"]
        chat_id = case.get("chat_id", 0)
        question = case["question"]

        context = retrieve_context.invoke({
            "name_or_email": email,
            "chat_id": chat_id,
            "query": question,
        })
        return str(context)

    # ---- Caso 2: RAG de BD (search_in_db) ----
    elif mode == "db":
        project_id = case["project_id"]
        question = case["question"]

        context = search_in_db.invoke({
            "project_id": project_id,
            "team_id": case.get("team_id"),
            "query": question,
            "max_tables": case.get("max_tables", 3),
            "k_per_table": case.get("k_per_table", 4),
            "only_tables": case.get("only_tables"),
        })
        return str(context)

    else:
        raise ValueError(f"Modo de caso desconocido: {mode}")


def judge_context(
    question: str,
    context: str,
    expected_answer: str,
) -> Dict[str, Any]:
    """
    Pide al LLM-juez que califique el contexto.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        expected=expected_answer,
    )

    resp = judge_llm.invoke(prompt)
    raw = resp.content if hasattr(resp, "content") else str(resp)

    try:
        data = json.loads(raw)
    except Exception:
        # fallback: intenta limpiar cosas raras
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
        except Exception:
            data = {"score": 1, "justificacion": f"JSON inválido del juez: {raw}"}

    # Normalizar score
    try:
        score = int(data.get("score", 1))
    except Exception:
        score = 1

    score = max(1, min(5, score))
    data["score"] = score
    return data


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Corre todo el pipeline de evaluación para un caso:
    - llama al RAG
    - pide al juez que calibre el contexto
    - devuelve resultados enriquecidos
    """
    case_id = case["id"]
    print(f"\n=== Evaluando caso: {case_id} ===")

    context = get_rag_context(case)
    judge_result = judge_context(
        question=case["question"],
        context=context,
        expected_answer=case["expected_answer"],
    )

    result = {
        "id": case_id,
        "mode": case["mode"],
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "rag_context_preview": context[:800],  # recorta por si es muy largo
        "score": judge_result["score"],
        "justificacion": judge_result.get("justificacion", ""),
    }

    print(f"Score: {result['score']} / 5")
    print(f"Justificación juez: {result['justificacion']}")
    return result


# ============================
# 4) Main
# ============================

def main():
    if not TEST_CASES:
        print("No hay TEST_CASES definidos. Edita rag_eval.py y agrega algunos.")
        return

    results: List[Dict[str, Any]] = []

    for case in TEST_CASES:
        try:
            res = evaluate_case(case)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Caso {case['id']}: {e}")

    if not results:
        print("No se pudo evaluar ningún caso.")
        return

    scores = [r["score"] for r in results]
    avg = statistics.mean(scores)
    print("\n========================")
    print("  RESUMEN RAG EVAL")
    print("========================")
    print(f"Casos evaluados: {len(results)}")
    print(f"Promedio score: {avg:.2f} / 5")

    # Guardar en JSONL para análisis posterior
    out_path = "rag_eval_results.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResultados guardados en: {out_path}")


if __name__ == "__main__":
    main()
