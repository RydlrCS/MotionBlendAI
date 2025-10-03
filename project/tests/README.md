# Running tests

This folder contains unit and integration-style tests for the MotionBlendAI project.

Quick local steps (from repository root):

```bash
# activate your virtualenv (example)
source .venv/bin/activate

# install dependencies
python -m pip install --upgrade pip
pip install -r project/elastic_search/requirements.txt
pip install pytest

# run the test suite
pytest -q project/tests/
```

Notes:
- The tests are designed to run without Docker by using dependency injection (the seeder test uses a DummyES client).
- To perform a full end-to-end run against Elasticsearch, start Docker Desktop and run `docker compose up` then run the seeder without `--dry-run`.
