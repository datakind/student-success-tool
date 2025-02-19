Notes: 

REST API for SST functionality.

Notes:
### API Callers

API callers will need to create a user using the backend and then generate an API token. They will also need the GCloud upload auth token.

### Prerequisites

In order to work with and test GCS related functionality, you'll need to setup default credentials:
https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment#local-user-cred

You will also need to add the permission Storage Writer or Storage Admin to your Datakind Account in GCP to allow for local interaction with the storage buckets.

Note that to generate GCP URLS you'll need a service account key (doesn't work locally).

### For local testing:

Enter into the root directory of the repo.


1. Copy the `.env.example` file to `.env`
1. Run `export ENV_FILE_PATH='/full/path/to/.env'`
1. `python3 -m venv .venv`
1. `source .venv/bin/activate`
1. `pip install uv`
1. `uv sync --all-extras --dev`
1. `coverage run -m pytest  -v -s ./src/webapp/`

For integration wtih Databricks, run:

The workspace URL will look like `https://<some_id>.gcp.databricks.com`

1. `databricks auth login --host <workspace_url>`

For all of the following, be in the repo root folder (`student-success-tool/`).

Spin up the app locally:

1. `fastapi dev src/webapp/main.py`
1. Go to `http://127.0.0.1:8000/docs`
1. Hit the `Authorize` button on the top right and enter the tester credentials:

* username: `tester@datakind.org`
* password: `tester_password`

Before committing, make sure to run:

1. `black src/webapp/.`
1. Test using `coverage run -m pytest  -v -s ./src/webapp/*.py`
1. Test using `coverage run -m pytest  -v -s ./src/webapp/routers/*.py`

### Notes:

postgresql requires that SSL certs be 0600 or 0640 depending on group/owners. The way we configure the 

The flow to add a Datakinder user is different from adding a user to an institution:
* adding a user to an institution has to happen prior to that user creating an account (by allowlisting their email for a given institution)
* adding a Datakinder user has to happen after the Datakinder person has already created their account, then their account's access type is updated.

In general, the service account used to run this service in GCP will also need to be granted Databricks access in the equivalent environment.

