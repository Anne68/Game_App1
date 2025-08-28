install: ; pip install -r requirements.txt
run: ; uvicorn api_games_plus:app --reload
test: ; pytest -q --maxfail=1 --disable-warnings
lint: ; flake8 api_games_plus.py settings.py
docker: ; docker build -t api-game .
