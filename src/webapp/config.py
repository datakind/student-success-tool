"""Helper dict to retrieve OS env variables. This list includes all environment variables needed.
"""

import os
from dotenv import load_dotenv

# defaults to unit test values.
env_vars = {
    "ENV": "LOCAL",
    "SECRET_KEY": "",
    "ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": "120",
    # The Issuers env var will be stored as an array of emails.
    "API_KEY_ISSUERS": [],
}

# The INSTANCE_HOST is the private IP of CLoudSQL instance e.g. '127.0.0.1' ('172.17.0.1' if deployed to GAE Flex)
engine_vars = {
    "INSTANCE_HOST": "",
    "DB_USER": "",
    "DB_PASS": "",
    "DB_NAME": "",
    "DB_PORT": "",
}

# For deployments that connect directly to a Cloud SQL instance without
# using the Cloud SQL Proxy, configuring SSL certificates will ensure the
# connection is encrypted.
# root cert: e.g. '/path/to/server-ca.pem'
# cert: e.g. '/path/to/client-cert.pem'
# key: e.g. '/path/to/client-key.pem'
ssl_env_vars = {
    "DB_ROOT_CERT": "",
    "DB_CERT": "",
    "DB_KEY": "",
}

gcs_vars = {
    "GCP_REGION": "",
    "GCP_SERVICE_ACCOUNT_EMAIL": "",
}

# Frontend vars needed for Laravel integration.
fe_vars = {
    # SECRET.
    "FE_USER": "fe-usr",
    "FE_HASHED_PASSWORD": "",
}

# databricks vars needed for databricks integration
databricks_vars = {
    # SECRET.
    "CATALOG_NAME": "",
    "DATABRICKS_WORKSPACE": "",
    "DATABRICKS_HOST_URL": "",
    # The service account that is used in Databricks to access GCP buckets.
    "DATABRICKS_SERVICE_ACCOUNT_EMAIL": "",
}


# Setup function to get environment variables. Should be called at startup time.
def startup_env_vars():
    env_file = os.environ.get("ENV_FILE_PATH")
    if not env_file:
        raise ValueError(
            "Missing .env filepath variable. Required. Set ENV_FILE_PATH to full path of .env file."
        )
    load_dotenv(env_file)
    global env_vars
    for name in env_vars:
        env_var = os.environ.get(name)
        if name == "API_KEY_ISSUERS":
            # This is okay to be empty, though slightly unexpected, it shouldn't fail.
            if not env_var:
                continue
            emails = env_var.split(",")
            env_vars[name] = [x.strip() for x in emails]
        if not env_var:
            raise ValueError(
                "Missing " + name + " value missing. Required environment variable."
            )
        if name == "ENV" and env_var not in [
            "PROD",
            "STAGING",
            "DEV",
            "LOCAL",
        ]:
            raise ValueError(
                "ENV environment variable not one of: PROD, STAGING, DEV, LOCAL."
            )
        if (
            name == "ACCESS_TOKEN_EXPIRE_MINUTES"
            or name == "ACCESS_TOKEN_EXPIRE_MINUTES"
        ) and not env_var.isdigit():
            raise ValueError(
                "ACCESS_TOKEN_EXPIRE_MINUTES and ACCESS_TOKEN_EXPIRE_MINUTES environment variables must be an int."
            )
        env_vars[name] = env_var
    if env_vars["ENV"] != "LOCAL":
        global gcs_vars
        for name in gcs_vars:
            env_var = os.environ.get(name)
            if not env_var:
                raise ValueError(
                    "Missing "
                    + name
                    + " value missing. Required GCP environment variable."
                )
            gcs_vars[name] = env_var
        global fe_vars
        for name in fe_vars:
            env_var = os.environ.get(name)
            if not env_var or env_var == "":
                raise ValueError(
                    "Missing "
                    + name
                    + " value missing. Required Frontend integration environment variable."
                )
            fe_vars[name] = env_var
        global databricks_vars
        for name in databricks_vars:
            env_var = os.environ.get(name)
            if not env_var or env_var == "":
                raise ValueError(
                    "Missing "
                    + name
                    + " value missing. Required Databricks integration environment variable."
                )
            databricks_vars[name] = env_var


# Setup function to get db environment variables. Should be called at db startup time.
def setup_database_vars():
    global engine_vars
    for name in engine_vars:
        env_var = os.environ.get(name)
        if not env_var:
            raise ValueError("Missing " + name + " value missing. Required.")
        engine_vars[name] = env_var

    if env_vars["ENV"] in ("LOCAL"):
        # doesn't require ssl vars
        return

    global ssl_env_vars
    for name in ssl_env_vars:
        env_var = os.environ.get(name)
        if not os.environ.get(name):
            raise ValueError(
                "Missing " + name + " value missing. Required for SSL connection."
            )
        ssl_env_vars[name] = env_var
