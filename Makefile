.PHONY: setup data test api dashboard docker-up clean

setup:
	pip install -r requirements.txt

data:
	python scripts/generate_cmapss_data.py --out data/raw

test:
	pytest tests/ -v --tb=short

api:
	uvicorn src.serving.api:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

docker-up:
	docker-compose up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache
