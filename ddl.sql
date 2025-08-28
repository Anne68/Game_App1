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
