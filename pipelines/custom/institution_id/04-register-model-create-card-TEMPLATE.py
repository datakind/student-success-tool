# Databricks notebook source
# MAGIC %md
# MAGIC # SST Register Model and Create Model Card
# MAGIC
# MAGIC Fifth step in the process of transforming raw data into actionable, data-driven insights for advisors: finalize model with unity catalog model registration and generate model card for transparency and reporting
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# WARNING: AutoML/mlflow expect particular packages with version constraints
# that directly conflicts with dependencies in our SST repo. As a temporary fix,
# we need to manually install a certain version of pandas and scikit-learn in order
# for our models to load and run properly.

# %pip install "student-success-tool==0.3.7"
# %pip install "pandas==1.5.3"
# %pip install "scikit-learn==1.3.0"
# %restart_python

# COMMAND ----------

import os
import mlflow
import logging
from databricks.connect import DatabricksSession

from student_success_tool import dataio, configs, modeling

# TODO for Vish: implement CustomModelCard
# from student_success_tool.reporting.model_card.pdp import PDPModelCard

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# HACK: hardcode uc base path and mlflow client
# NOTE: registry uri needs to be set before creating the client
# to avoid mlflow REST exception when registering the model
catalog = "sst_dev"
mlflow.set_registry_uri("databricks-uc")
client = mlflow.tracking.MlflowClient()

# HACK: We need to disable the mlflow widget template loading for MC output
# Retrieved from DB office hours, otherwise 10+ DB widgets try to load and
# fail when pulling from ML artifacts (it's annoying)
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

# COMMAND ----------

# MAGIC %md
# MAGIC ## import school-specific code

# COMMAND ----------

# project configuration stored as a config file in TOML format
cfg = dataio.read_config(
    "./config-TEMPLATE.toml", schema=configs.custom.CustomProjectConfig
)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # register model

# COMMAND ----------

model_name = modeling.registration.get_model_name(
    institution_id=cfg.institution_id,
    target=cfg.preprocessing.target.name,
    checkpoint=cfg.preprocessing.checkpoint.name,
)
model_name

# COMMAND ----------

modeling.registration.register_mlflow_model(
    model_name,
    cfg.institution_id,
    run_id=cfg.model.run_id,
    catalog=catalog,
    registry_uri="databricks-uc",
    mlflow_client=client,
)
