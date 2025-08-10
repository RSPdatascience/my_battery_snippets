import pyodbc
import warnings
import pandas as pd

def query_to_dataframe(sql_string, conn_str):
    try:
        conn = pyodbc.connect(conn_str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            df = pd.read_sql(sql_string, conn)
        conn.close()
        return df
    except Exception as e:
        print("Query failed:", e)
        return pd.DataFrame()