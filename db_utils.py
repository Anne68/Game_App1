import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def get_engine():
    load_dotenv()
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "3306")
    db   = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASSWORD")
    if not all([host, db, user, pwd]):
        raise RuntimeError("Missing DB_* environment variables")
    url = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)

def ensure_unique_keys():
    ddl = """
    ALTER TABLE games
      ADD UNIQUE KEY uq_games_rawg (game_id_rawg),
      ADD UNIQUE KEY uq_games_title (title);
    ALTER TABLE platforms
      ADD UNIQUE KEY uq_platforms_id (platform_id),
      ADD UNIQUE KEY uq_platforms_name (platform_name);
    ALTER TABLE best_price_pc
      ADD UNIQUE KEY uq_best_price_title (title);
    ALTER TABLE game_platforms
      ADD UNIQUE KEY uq_gp (game_id_rawg, platform_id);
    """
    eng = get_engine()
    with eng.begin() as c:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            try:
                c.execute(text(stmt))
            except Exception as e:
                print("[WARN] DDL:", e)

def _table_columns(engine, table):
    with engine.begin() as c:
        cols = pd.read_sql(text(f"SHOW COLUMNS FROM `{table}`"), c)["Field"].tolist()
    return cols

def upsert_existing_and_insert_new(df, table, key_col, date_field=None, new_limit=50):
    if df is None or df.empty:
        print(f"[SKIP] {table}: dataframe vide")
        return 0, 0

    engine = get_engine()
    cols_db = _table_columns(engine, table)

    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    keep = [c for c in df.columns if c in cols_db]
    if key_col not in keep and key_col in df.columns:
        keep.append(key_col)
    df = df[keep]

    if key_col not in df.columns:
        raise ValueError(f"Colonne clé '{key_col}' absente pour {table}")

    with engine.begin() as c:
        existing = pd.read_sql(text(f"SELECT `{key_col}` AS k FROM `{table}`"), c)["k"]

    df["_key_str_"] = df[key_col].astype(str)
    existing_str = set(existing.astype(str))

    df_exist = df[df["_key_str_"].isin(existing_str)].copy()
    df_cand  = df[~df["_key_str_"].isin(existing_str)].copy()

    if date_field and date_field in df_cand.columns:
        df_cand["_dt_"] = pd.to_datetime(df_cand[date_field], errors="coerce")
        df_cand = df_cand.sort_values("_dt_", ascending=False).drop(columns=["_dt_"])

    df_new = df_cand.head(new_limit)

    def _bulk_upsert(df_part):
        if df_part.empty:
            return 0
        cols = [c for c in df_part.columns if c != "_key_str_"]
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(f"`{c}`" for c in cols)
        update_cols = [c for c in cols if c != key_col]
        updates = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in update_cols]) or f"`{key_col}`=`{key_col}`"
        sql = f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
        data = df_part[cols].where(pd.notnull(df_part), None).values.tolist()
        with engine.begin() as conn:
            conn.execute(text("SET SESSION sql_mode=''"))
            conn.execute(text(sql), data)
        return len(df_part)

    n_upd = _bulk_upsert(df_exist)
    n_new = _bulk_upsert(df_new)
    print(f"[OK] {table}: updated={n_upd}, inserted={n_new} (limit {new_limit})")
    return n_upd, n_new
