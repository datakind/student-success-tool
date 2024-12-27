"""Helper dict to retrieve OS env variables. This list includes all environment variables needed.
"""

import os
from dotenv import load_dotenv


# Setup function to get environment variables. Should be called at startup time.
def startup_env_vars():
    env_file = os.environ.get("ENV_FILE_PATH")
    if not env_file:
        raise ValueError(
            "Missing .env filepath variable. Required. Set ENV_FILE_PATH to full path of .env file."
        )
    load_dotenv(env_file)
    env = os.environ.get("ENV")
    if not env:
        raise ValueError(
            "Missing ENV environment variable. Required. Can be PROD, STAGING, DEV, or LOCAL."
        )
    if env not in ["PROD", "STAGING", "DEV", "LOCAL"]:
        raise ValueError(
            "ENV environment variable not one of: PROD, STAGING, DEV, or LOCAL."
        )


env_vars = {
    "ENV": os.environ.get("ENV"),
}

# The INSTANCE_HOST is the private IP of CLoudSQL instance e.g. '127.0.0.1' ('172.17.0.1' if deployed to GAE Flex)
engine_vars = {
    "INSTANCE_HOST": os.environ.get("INSTANCE_HOST"),
    "DB_USER": os.environ.get("DB_USER"),
    "DB_PASS": os.environ.get("DB_PASS"),
    "DB_NAME": os.environ.get("DB_NAME"),
    "DB_PORT": os.environ.get("DB_PORT"),
}

# For deployments that connect directly to a Cloud SQL instance without
# using the Cloud SQL Proxy, configuring SSL certificates will ensure the
# connection is encrypted.
# root cert: e.g. '/path/to/server-ca.pem'
# cert: e.g. '/path/to/client-cert.pem'
# key: e.g. '/path/to/client-key.pem'
ssl_env_vars = {
    "DB_ROOT_CERT": os.environ.get("DB_ROOT_CERT"),
    "DB_CERT": os.environ.get("DB_CERT"),
    "DB_KEY": os.environ.get("DB_KEY"),
}
