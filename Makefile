install:
	pip install -r requirements.txt
run:
	uvicorn app.main:app --reload
test:
	pytest -q --maxfail=1 --disable-warnings
lint:
	flake8 app
docker:
	docker build -t api-game .
