# Databricks notebook source
# MAGIC %md
# MAGIC # SST Train and Evaluate Model: [SCHOOL]
# MAGIC
# MAGIC Third step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: load a prepared modeling dataset, configure experiment tracking framework, then train and evaluate a predictive model.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Classification with AutoML](https://docs.databricks.com/en/machine-learning/automl/classification.html)
# MAGIC - [Databricks AutoML Python API reference](https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - TODO: [SCHOOL] website

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, most of which should come through our 1st-party SST package
# %pip install "student-success-tool==0.1.0"
# %pip install git+https://github.com/datakind/student-success-tool.git@develop

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging

import mlflow
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils

from student_success_tool import modeling
from student_success_tool.analysis import pdp

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark_session = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

run_parameters = dict(dbutils.notebook.entry_point.getCurrentBindings())
job_run_id = run_parameters.get("job_run_id", "interactive")

# COMMAND ----------

# TODO: specify school-specific configuration
institution_id = "SCHOOL"
table_name = "CATALOG.SCHEMA.TABLE"
student_id_col = "student_guid"
target_col = "target"
student_group_cols = [
    "student_age",
    "race",
    "ethnicity",
    "gender",
    "first_gen",
]
optional_automl_parameters = {
    "split_col": "split",
    "sample_weight_col": "sample_weight",
    # "pos_label": True,
    # exclude_frameworks: ["lightgbm", "xgboost"],
    "timeout_minutes": 5,
}
optimization_metric = "log_loss"

prediction_col = "prediction"
risk_score_col = "risk_score"

optional_automl_parameters["exclude_cols"] = list(
    set(optional_automl_parameters.get("exclude_cols", []) + student_group_cols)
)
optional_automl_parameters

# COMMAND ----------

# MAGIC %md
# MAGIC # read modeling dataset

# COMMAND ----------

df = pdp.schemas.PDPLabeledDataSchema(
    pdp.dataio.read_data_from_delta_table(table_name, spark_session=spark_session)
)
print(f"rows x cols = {df.shape}")
df.head()

# COMMAND ----------

if split_col := optional_automl_parameters.get("split_col"):
    print(df[split_col].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC # train model

# COMMAND ----------

summary = modeling.training.run_automl_classification(
    df,
    target_col=target_col,
    optimization_metric=optimization_metric,
    institution_id=institution_id,
    job_run_id=job_run_id,
    student_id_col=student_id_col,
    **optional_automl_parameters,
)

experiment_id = summary.experiment.experiment_id
experiment_run_id = summary.best_trial.mlflow_run_id
print(
    f"experiment_id: {experiment_id}"
    f"\n{optimization_metric} metric distribution = {summary.metric_distribution}"
    f"\nbest trial experiment_run_id: {experiment_run_id}"
)

dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
dbutils.jobs.taskValues.set(key="experiment_run_id", value=experiment_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # evaluate model

# COMMAND ----------

# MAGIC %md
# MAGIC **TODO:** This doesn't currently work, owing to version incompatibilities between Databricks AutoML runtime and the `student-success-tool` package. Revisit this once we're using our own "auto"-ML framework. Look to existing nb for guidance on additional evaluation needs.

# COMMAND ----------

# mlflow.sklearn.load_model(summary.best_trial.model_path)
model = summary.best_trial.load_model()
model

# COMMAND ----------
