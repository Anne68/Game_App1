# Exemple côté backend, après avoir obtenu la liste recos:
# for rec in recommendations: rec["id"] = game_id RAWG/DB

# Pseudo-code (dans l’API):
with get_db_conn() as conn, conn.cursor() as cur:
    ids = tuple([r["id"] for r in recommendations])
    cur.execute("""
        SELECT game_id_rawg AS id, price, store AS store, store_url AS url
        FROM games
        WHERE game_id_rawg IN %s
    """, (ids,))
    rows = {int(r["id"]): r for r in cur.fetchall()}

for r in recommendations:
    info = rows.get(int(r["id"]), {})
    r["price"] = info.get("price")
    r["store"] = info.get("store")
    r["url"]   = info.get("url")
