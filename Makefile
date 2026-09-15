# LeLamp affect pipeline -- everything a reviewer needs, in four targets.
PY := .venv/bin/python
export PYTHONPATH := src

.DEFAULT_GOAL := help

help:
	@echo "make setup      create .venv and install dependencies (~2 min)"
	@echo "make demo       stream one MELD dialogue through the lamp, live  <-- start here"
	@echo "make grounding  counterfactual: same words, different video"
	@echo "make eval       regenerate every number and figure (~2 min, cached features)"
	@echo "make ledger     print the parameter ledger"
	@echo "make test       run the unit tests"
	@echo ""
	@echo "make features   RE-DERIVE features by streaming MELD.Raw (10.9 GB, ~25 min)"
	@echo "make train      retrain the heads from features (~1 min, CPU)"

setup:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	@echo "optional local LLM responder:  $(PY) -m pip install mlx-lm"

demo:
	$(PY) scripts/demo.py --dialogue 237

grounding:
	$(PY) scripts/demo.py --counterfactual

eval:
	$(PY) -m lelamp.evaluate.run_all
	$(PY) scripts/render_results.py

ledger:
	$(PY) -m lelamp.ledger

test:
	$(PY) -m pytest tests/ -q

# --- heavy, optional: regenerates the cached features from the raw dataset ---
# Order matters: text first (cheap, no network), then the solo prefixes it
# extends, then the 10.9 GB video stream. Run sequentially -- two processes
# using Metal at once wedges one of them on Apple Silicon (see README).
features:
	$(PY) scripts/extract_text.py
	$(PY) scripts/extract_solo_prefix.py
	$(PY) -m lelamp.data.stream_extract --train-cap 0 --vendor "$$(cat assets/DEMO_UIDS.txt)"
	$(PY) -m lelamp.train

train:
	$(PY) -m lelamp.train

clean:
	rm -rf artifacts/figures/*.png artifacts/runs/*.jsonl artifacts/results.json

.PHONY: help setup demo grounding eval ledger test features train clean
