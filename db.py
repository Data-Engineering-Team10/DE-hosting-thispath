import streamlit as st
import base64
import numpy as np
import pandas as pd
import psycopg2


@st.cache_resource
def fetch_wines_embeddings():
    df_wine = select_table('wines')
    df_embedding = df_wine[['embeddings']].iloc[:, 0]
    df_embedding = np.stack(df_embedding)
    return df_wine, df_embedding
    
@st.cache_resource
def init_connection():
    """ Initialize connection. Uses st.cache_resource to only run once

    Returns:
        (Connect): database connection
    """
    return psycopg2.connect(**st.secrets["postgres"])

def select_table(table_name, column_list=None, where_dict=None, order_by=None):
    """ Perform query, return its query and columns

    Args:
        table_name (str): database table name
        column_list (list, optional): columns for query. Defaults to None.
        where_dict (dict, optional): {column name: value} Defaults to None.
        order_by (str, optional): ordering strategy. Defaults to None.

    Returns:
        query_result (pd.DataFrame): query result.
    """
    if column_list is None:
        column_clause = "*"
    else:
        column_clause = ", ".join(column_list)

    query = f"SELECT {column_clause} FROM {table_name}"
    values = []

    if where_dict is not None:
        where_clause = " AND ".join([f"{k} = %s" for k in where_dict.keys()])
        query += f" WHERE {where_clause}"
        values = list(where_dict.values())

    if order_by is not None:
        query += f" ORDER BY {order_by}"

    query_result = run_query(query, values)
    return query_result

def update_table(table_name, update_dict, where_dict):
    """ Update database table

    Args:
        table_name (str): database table name
        update_dict (dict): {column name: value}
        where_dict (dict): {column name: value}

    Returns:
        query_result (pd.DataFrame): query result.
    """
    update_clause = ", ".join([f"{k} = %s" for k in update_dict.keys()])
    where_clause = " AND ".join([f"{k} = %s" for k in where_dict.keys()])
    query = f"UPDATE {table_name} SET {update_clause} WHERE {where_clause}"
    values = list(update_dict.values()) + list(where_dict.values())
    query += " RETURNING *"
    
    query_result = run_query(query, values)
    return query_result

def insert_table(table_name, row_dict):
    """ Insert data into database table

    Args:
        table_name (str): database table name
        row_dict (dict): {column name: value}

    Returns:
        query_result (pd.DataFrame): query result.
    """
    columns_clause = ', '.join(row_dict.keys())
    values_clause = ', '.join(['%s'] * len(row_dict))
    query = f"INSERT INTO {table_name} ({columns_clause}) VALUES ({values_clause})"
    values = tuple(row_dict.values())
    query += " RETURNING *"

    query_result = run_query(query, values)
    return query_result

def run_query(query, values):
    """ Run query. If exception occurs, then raise exception once again
    to detect query is effective.

    Args:
        query (str): query clause
        values (Sequence): values

    Raises:
        e (Exception): If exception occurs, then raise exception once again.

    Returns:
        query_result (pd.DataFrame): query result.
    """
    with init_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)
            results = cur.fetchall()
            if len(results) == 0:
                raise Exception
            
            columns = [desc[0] for desc in cur.description]
            query_result = pd.DataFrame(results, columns=columns)
            if 'embeddings' in query_result:
                query_result['embeddings'] = query_result['embeddings'].apply(decode_vector)
        return query_result

def decode_vector(string_vec):
    """ Decode serialized vector into real number vector

    Args:
        string_vec (str): serialized vector

    Returns:
        np.array: real number vector
    """
    embeddings = base64.b85decode(string_vec)
    embeddings = np.frombuffer(embeddings, dtype=np.float32)
    return embeddings

def encode_vector(vector):
    """ Encode real number vector into serialized vector

    Returns:
        np.array: serialized vector
    """
    embeddings = np.float32(vector).tobytes()
    embeddings = base64.b85encode(embeddings).decode()
    return embeddings
