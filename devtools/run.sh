#!/usr/bin/env bash

source devtools/lib.sh || { echo "Run this tool from repo root."; exit 1; }

# Usage: ./devtools/run.sh [options]
# Run the docker-compose or podman-compose command with the given options. The
# options are passed directly to the compose command. See the compose command
# documentation for more information, e.g. `podman compose --help`.

# This script will use podman if it is installed, otherwise it will use docker.
# To install podman, see https://podman.io/getting-started/installation.html
# To install docker, see https://docs.docker.com/get-docker/

args="$@"
if [ -z "$args" ]; then
  args="up"
fi

if command -v podman 2>&1 >/dev/null
then
  runcmd podman compose $args
elif command -v docker 2>&1 >/dev/null
then
  runcmd docker compose $args
else
  echo "Neither podman nor docker is installed"
  exit 1
fi

