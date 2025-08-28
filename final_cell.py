from scripts.db_utils import ensure_unique_keys, upsert_existing_and_insert_new

ensure_unique_keys()

upsert_existing_and_insert_new(
    df=df_games,
    table="games",
    key_col="game_id_rawg",
    date_field="release_date",
    new_limit=50
)

upsert_existing_and_insert_new(
    df=df_platforms,
    table="platforms",
    key_col="platform_id",
    date_field=None,
    new_limit=10**9
)

upsert_existing_and_insert_new(
    df=df_best_price,
    table="best_price_pc",
    key_col="title",
    date_field="last_update",
    new_limit=10**9
)

upsert_existing_and_insert_new(
    df=df_game_platforms,
    table="game_platforms",
    key_col="game_id_rawg",
    date_field=None,
    new_limit=10**9
)

print("[OK] Données mises à jour + 50 nouveaux jeux insérés.")
