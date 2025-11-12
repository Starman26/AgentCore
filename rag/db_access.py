from supabase import create_client, Client
import pandas as pd
import os

DB : Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

def retrieve_robot_support() -> pd.DataFrame:
    res = DB.table("RoboSupportDB").select(
        "created_at, robot_type, problem_title, problem_description, solution_steps, author"
    ).execute()
    
    rows = res.data
    df = pd.DataFrame(rows)
    return df
    