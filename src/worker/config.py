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
    "USERNAME": "tester-user",
    "PASSWORD": "tester-pw",
    "BACKEND_API_KEY": "",
}

gcs_vars = {
    "GCP_SERVICE_ACCOUNT_KEY_PATH": "",
}

sftp_vars = {
    "SFTP_HOST": "",
    "SFTP_PORT": "",
    "SFTP_USER": "",
    "SFTP_PASSWORD": "",
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
        global sftp_vars
        for name in sftp_vars:
            env_var = os.environ.get(name)
            if not env_var:
                raise ValueError(
                    "Missing "
                    + name
                    + " value missing. Required SFTP environment variable."
                )
            sftp_vars[name] = env_var
