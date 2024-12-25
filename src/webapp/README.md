Notes: 

REST API for SST functionality.


For local testing:

Enter into the root directory of the repo.

1. `python3 -m venv .venv`
1. `source .venv/bin/activate`
1. `pip install uv`
1. `uv sync --all-extras --dev`
1. `coverage run -m pytest  -v -s ./src/webapp/`