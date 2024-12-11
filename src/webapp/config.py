"""Helper dict to retrieve OS env variables. This list includes all environment variables needed.
"""

import os

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
