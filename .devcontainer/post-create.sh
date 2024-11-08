#!/usr/bin/env bash

source devtools/lib.sh

# Install dependencies
runcmd uv sync --frozen --no-install-project --no-dev

# Install gcloud cli
runcmd gcloud init --skip-diagnostics

runcmd gcloud auth application-default login --impersonate-service-account local-webapp@dev-sst-439514.iam.gserviceaccount.com