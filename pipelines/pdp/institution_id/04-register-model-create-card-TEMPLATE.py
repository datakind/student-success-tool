# Databricks notebook source
# MAGIC %md
# MAGIC # SST Register Model and Create Model Card: [SCHOOL]
# MAGIC
# MAGIC Fifth step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: finalize model with unity catalog model registration and generate model card for transparency and reporting
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [SCHOOL WEBSITE](https://example.com)

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

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os
import mlflow
import logging
from databricks.connect import DatabricksSession

from student_success_tool import dataio, configs, modeling
from student_success_tool.reporting.model_card.pdp import PDPModelCard

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

# project configuration should be stored in a config file in TOML format
# it'll start out with just basic info: institution_id, institution_name
# but as each step of the pipeline gets built, more parameters will be moved
# from hard-coded notebook variables to shareable, persistent config fields
cfg = dataio.read_config("./config-TEMPLATE.toml", schema=configs.pdp.PDPProjectConfig)
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

# COMMAND ----------

# MAGIC %md
# MAGIC # generate model card

# COMMAND ----------

# Initialize card
card = PDPModelCard(
    config=cfg, catalog=catalog, model_name=model_name, mlflow_client=client
)

# COMMAND ----------

# Build context and download artifacts
card.build()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Edit model card markdown file as you see fit before exporting as a PDF
# MAGIC - A markdown should now exist in your local directory. Feel free to edit directly in DB's text editor before running the cell below.
# MAGIC - You don't need to refresh the browser or restart your cluster etc, the model card function will re-read the markdown below before exporting as a PDF
# MAGIC - You can access the PDF via ML artifacts in your registered model, as you will not be able to view the PDF locally in DB workspace.
# MAGIC

# COMMAND ----------

# Reload & publish
card.reload_card()
card.export_to_pdf()
