from supabase import create_client, Client
from langchain_core.documents import Document
import pandas as pd
import os

SB : Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def _fetch_student(name_or_email: str):
    q = name_or_email.strip()
    if "@" in q:
        res = SB.table("students").select("*").eq("email", q).limit(1).execute()
    else:
        res = SB.table("students").select("*").ilike("full_name", f"%{q}%").limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None

def retrieve_robot_support() -> pd.DataFrame:
    res = SB.table("RoboSupportSB").select(
        "created_at, robot_type, problem_title, problem_description, solution_steps, author"
    ).execute()

    rows = res.data
    df = pd.DataFrame(rows)
    return df

def retrieve_student_info(name_or_email):
    row = _fetch_student(name_or_email)
    docs = []

    if not row:
        return [], 0

    metadata = {
        "id": row.get("id"),
        "full_name": row.get("full_name"),
        "email": row.get("email"),
        "major": row.get("major")
    }
    content = f"""Skills: {row.get('skills')}
                    Goals: {row.get('goals')}
                    Interests: {row.get('interests')}
                    Learning Style: {row.get('learning_style')}"""

    docs.append(Document(page_content=content, metadata=metadata))
    return docs, len(docs)

def retrieve_chat_summary(chat_id):
    res = SB.table("chat_summary").select(
        "id, session_id, summary_md, updated_at"
    ).eq("id", chat_id).execute()

    rows = res.data or []
    docs = []

    if not rows:
        return docs, 0

    row = rows[0]
    metadata = {
        "id": row.get("id"),
        "session_id": row.get("session_id"),
        "updated_at": row.get("updated_at"),
    }
    content = f"""
                summary_md: {row.get('summary_md')}\n 
                """
    docs.append(Document(page_content=content, metadata=metadata))

    return docs, len(docs)
