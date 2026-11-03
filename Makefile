PY := uv run python

.PHONY: all setup data features train baseline test analyze report site clean

all: setup data features train baseline  ## everything, from a fresh clone

setup:  ## install dependencies into .venv
	uv sync

data:  ## download the warehouse snapshot + upcoming requests (130 MB)
	$(PY) -m donor_targeting.download_data

features:  ## the DS team's feature sample -> data/warehouse/features.parquet
	$(PY) -m donor_targeting.features

train:  ## the existing model -> models/
	$(PY) -m donor_targeting.train

baseline:  ## BAU vs centroid precision/recall -> reports/baseline_report.md
	$(PY) -m donor_targeting.baseline

analyze:  ## churn analysis + cadence + box validation + forest runs + 3D embeddings + report
	$(PY) experiments/churn_analysis.py
	$(PY) experiments/cadence_analysis.py
	$(PY) experiments/box_validation.py
	$(PY) experiments/model_runs.py
	$(PY) experiments/embeddings_3d.py
	$(PY) experiments/render_report.py

site:  ## serve the dashboard at http://localhost:8899
	$(PY) -m http.server 8899 --directory site

precompute:  ## nightly job: rebuild scoring artifacts from the warehouse
	$(PY) -m service.precompute

serve:  ## run the API locally (Render-worker parity: limit with cgroups/docker if needed)
	$(PY) -m uvicorn service.api:app --host 127.0.0.1 --port 8000

up:  ## one command, production-parity: 2 GB / 4 CPU container
	docker compose up --build

evidence:  ## replay the 12 requests + chaos tests; writes reports/SERVICE.md
	$(PY) experiments/service_evidence.py

test:
	uv run pytest -q

clean:
	rm -rf data models reports
