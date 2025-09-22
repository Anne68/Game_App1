def _render_reco_table_with_filters(recs: List[Dict[str, Any]], selected_platforms: List[str], min_price: float):
    if not recs:
        st.info("Aucune recommandation.")
        return

    df = pd.json_normalize(recs)

    # Normalisation des plateformes (liste)
    platforms_series = _normalize_platforms_col(df)

    # Détection colonne prix / store / url
    price_col = _detect_price_col(df)      # ex: price, price_eur, msrp...
    store_col = _detect_store_col(df)      # ex: store, shop, purchase_platform...
    url_col   = _detect_url_col(df)        # ex: url, store_url, purchase_url...

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # Filtres côté client
    if selected_platforms:
        mask = platforms_series.apply(lambda lst: any(p.lower() in [x.lower() for x in lst] for p in selected_platforms))
        df = df[mask]
    if min_price and price_col:
        df = df[df[price_col] >= float(min_price)]

    # Vue minimaliste: titre, prix, url (+ store si dispo)
    title_col = "title" if "title" in df.columns else None
    display_cols = []
    rename_map = {}

    if title_col:
        display_cols.append(title_col); rename_map[title_col] = "titre"
    if price_col:
        display_cols.append(price_col); rename_map[price_col] = "prix"
    if store_col:
        display_cols.append(store_col); rename_map[store_col] = "store"
    if url_col:
        display_cols.append(url_col); rename_map[url_col] = "url"

    # Si l'API ne renvoie ni prix ni url, on affiche un fallback
    if not display_cols:
        base_cols = [c for c in ["confidence","id","title","genres","rating","metacritic"] if c in df.columns]
        df_disp = df[base_cols].rename(columns={"id":"game_id","title":"titre"})
        st.dataframe(df_disp, use_container_width=True)
        st.info("Astuce: l'API ne renvoie pas encore de prix / URL de boutique. Ajoutez ces champs côté API pour les afficher ici.")
        return

    df_disp = df[display_cols].rename(columns=rename_map).copy()

    # Rendre le titre cliquable si une URL est dispo
    if "url" in df_disp.columns:
        def mk_link(row):
            u = str(row.get("url", "")).strip()
            t = str(row.get("titre", "")).strip() or "Voir"
            return f"[{t}]({u})" if u and u.startswith("http") else t
        df_disp["titre"] = df_disp.apply(mk_link, axis=1)

    st.dataframe(df_disp, use_container_width=True)

    # Liste compacte en dessous (pratique pour copier)
    if "url" in df_disp.columns:
        st.caption("Liens d'achat")
        items = []
        for _, r in df_disp.iterrows():
            parts = [r.get("titre")]
            if "prix" in df_disp.columns and pd.notna(r.get("prix")):
                parts.append(f"— {r.get('prix')}")
            if "store" in df_disp.columns and pd.notna(r.get("store")):
                parts.append(f"({r.get('store')})")
            items.append(" ".join([p for p in parts if p]))
        st.markdown("\n".join([f"- {x}" for x in items]))
