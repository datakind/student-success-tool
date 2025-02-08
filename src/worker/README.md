# REST API for SST peripheral jobs/actions.

There is no user database. This job has no access to the databases used by the SST.

### Prerequisites

In order to work with and test GCS related functionality, you'll need to setup default credentials:
https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment#local-user-cred

You will also need to add the permission Storage Writer or Storage Admin to your Datakind Account in GCP to allow for local interaction with the storage buckets.

### For local testing:

Enter into the root directory of the repo.


1. Copy the `.env.example` file to `.env`
1. Run `export ENV_FILE_PATH='/full/path/to/.env'`
1. `python3 -m venv .venv`
1. `source .venv/bin/activate`
1. `pip install uv`
1. `uv sync --all-extras --dev`
1. `coverage run -m pytest  -v -s ./src/webapp/`

For all of the following, be in the repo root folder (`student-success-tool/`).

Spin up the app locally:

1. `fastapi dev src/webapp/main.py`
1. Go to `http://127.0.0.1:8000/docs`
1. Hit the `Authorize` button on the top right and enter the tester credentials:

* username: `tester-user`
* password: `tester-pw`

Before committing, make sure to run:

1. `black src/webapp/.`
1. Test using `coverage run -m pytest  -v -s ./src/webapp/*.py`
1. Test using `coverage run -m pytest  -v -s ./src/webapp/routers/*.py`
