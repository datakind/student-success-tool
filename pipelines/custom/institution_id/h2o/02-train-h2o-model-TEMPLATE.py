# Databricks notebook source
# MAGIC %md
# MAGIC # SST Train and Evaluate H2O Model
# MAGIC
# MAGIC Third step in the process of transforming raw data into actionable, data-driven insights for advisors: load a prepared modeling dataset, configure experiment tracking framework, then train and evaluate a predictive model.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Classification with AutoML](https://docs.databricks.com/en/machine-learning/automl/classification.html)
# MAGIC - [Databricks AutoML Python API reference](https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# WARNING: AutoML/mlflow expect particular packages with version constraints
# that directly conflicts with dependencies in our SST repo. As a temporary fix,
# we need to manually install a certain version of pandas and scikit-learn in order
# for our models to load and run properly.

# %pip install git+https://github.com/datakind/student-success-tool.git@feat/h2o
# %restart_python

# COMMAND ----------

import logging
import os

import mlflow
from mlflow.tracking import MlflowClient
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from pyspark.dbutils import DBUtils

dbutils = DBUtils(spark)
client = MlflowClient()

from student_success_tool import configs, dataio, modeling, utils
from student_success_tool.modeling import h2o_modeling

import h2o

# HACK: Disable the mlflow widget template otherwise it freaks out
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

logging.info("Starting H2O cluster...")
h2o.init()

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    user_email = ctx.tags().get("user").get()
    if user_email:
        workspace_path = f"/Users/{user_email}"
        logging.info(f"retrieved workspace path at {workspace_path}")
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# Get job run id for automl run
job_run_id = utils.databricks.get_db_widget_param("job_run_id", default="interactive")

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

# project configuration stored as a config file in TOML format
cfg = dataio.read_config(
    "./config-TEMPLATE.toml", schema=configs.custom.CustomProjectConfig
)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # read preprocessed dataset

# COMMAND ----------

df = dataio.read.from_delta_table(
    cfg.datasets.silver["preprocessed"].table_path,
    spark_session=spark,
)
df.head()

# COMMAND ----------

# delta tables not great about maintaining dtypes; this may be needed
# df = df.convert_dtypes()

# COMMAND ----------

print(f"target proportions:\n{df[cfg.target_col].value_counts(normalize=True)}")

# COMMAND ----------

if cfg.split_col:
    print(f"split proportions:\n{df[cfg.split_col].value_counts(normalize=True)}")

# COMMAND ----------

if cfg.sample_weight_col:
    print(f"sample weights: {df[cfg.sample_weight_col].unique()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # feature selection

# COMMAND ----------

# databricks freaks out during feature selection if autologging isn't disabled :shrug:
mlflow.autolog(disable=True)

# COMMAND ----------

# load feature selection params from the project config
# HACK: set non-feature cols in params since it's computed outside
# of feature selection config
selection_params = cfg.modeling.feature_selection.model_dump()
selection_params["non_feature_cols"] = cfg.non_feature_cols
logging.info("selection params = %s", selection_params)

# COMMAND ----------

df_selected = modeling.feature_selection.select_features(
    (df.loc[df[cfg.split_col].eq("train"), :] if cfg.split_col else df),
    **selection_params,
)
print(f"rows x cols = {df_selected.shape}")
df_selected.head()

# COMMAND ----------

# HACK: we want to use selected columns for *all* splits, not just train
df = df.loc[:, df_selected.columns]

# COMMAND ----------

# save modeling dataset with all splits
dataio.write.to_delta_table(
    df, cfg.datasets.silver["modeling"].table_path, spark_session=spark
)

# COMMAND ----------

# MAGIC %md
# MAGIC # train model

# COMMAND ----------

training_params = {
    "job_run_id": job_run_id,
    "institution_id": cfg.institution_id,
    "student_id_col": cfg.student_id_col,
    "target_col": cfg.target_col,
    "split_col": cfg.split_col,
    "pos_label": cfg.pos_label,
    "primary_metric": "logloss",
    "timeout_minutes": cfg.modeling.training.timeout_minutes,
    "exclude_cols": sorted(
        set(
            (cfg.modeling.training.exclude_cols or [])
            + (cfg.student_group_cols or [])
            + (cfg.non_feature_cols or [])
        )
    ),
    "target_name": cfg.preprocessing.target.name,
    "checkpoint_name": cfg.preprocessing.checkpoint.name,
    "workspace_path": workspace_path,
    "seed": cfg.random_state,
}
logging.info("training params = %s", training_params)

# COMMAND ----------

experiment_id, aml, train, valid, test = (
    h2o_modeling.training.run_h2o_automl_classification(
        df=df,
        **training_params,
        client=client,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # evaluate model

# COMMAND ----------

# HACK: Evaluate an experiment you've already trained
# experiment_id = cfg.model.experiment_id

# NOTE: AutoML generates a split column if not manually specified
split_col = training_params.get("split_col", "_automl_split_col_0000")

# COMMAND ----------

# only possible to do bias evaluation if you specify a split col for train/test/validate
# AutoML doesn't preserve student ids the training set, which we need for [reasons]
if evaluate_model_bias := (training_params.get("split_col") is not None):
    df_features = df.drop(columns=cfg.non_feature_cols)
else:
    df_features = modeling.evaluation.extract_training_data_from_model(experiment_id)

# COMMAND ----------

# Get top runs from experiment for evaluation
# H2O metric names are slightly different TODO: Vish to maybe make them the same?
top_runs = modeling.evaluation.get_top_runs(
    experiment_id,
    optimization_metrics=[
        "test_recall",
        "test_roc_auc",
        "test_log_loss",
        "test_f1",
        "validate_log_loss",
    ],
    topn_runs_included=cfg.modeling.evaluation.topn_runs_included,
)
logging.info("top run ids = %s", top_runs)

# COMMAND ----------

for run_id in top_runs.values():
    with mlflow.start_run(run_id=run_id) as run:
        logging.info(
            "Run %s: Starting performance evaluation%s",
            run_id,
            " and bias assessment",
        )
        # load model and predict
        # features are already preprocessed (no need for imputer)
        model = h2o_modeling.utils.load_h2o_model(run_id=run_id)
        h2o_frame = h2o.H2OFrame(df_features)
        preds_df = model.predict(h2o_frame).as_data_frame()

        df_pred = df.assign(
            **{
                cfg.pred_col: preds_df["predict"].values,
                cfg.pred_prob_col: preds_df.iloc[:, 1].values,
            }
        )

        modeling.evaluation.evaluate_performance(
            df_pred,
            target_col=cfg.target_col,
            pos_label=cfg.pos_label,
        )

        modeling.bias_detection.evaluate_bias(
            df_pred,
            student_group_cols=cfg.student_group_cols,
            target_col=cfg.target_col,
            pos_label=cfg.pos_label,
        )
        logging.info("Run %s: Completed", run_id)
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC # model selection

# COMMAND ----------

# Rank top runs again after evaluation for model selection
selected_runs = modeling.evaluation.get_top_runs(
    experiment_id,
    optimization_metrics=[
        "test_recall",
        "test_roc_auc",
        "test_log_loss",
        "test_bias_score_mean",
    ],
    topn_runs_included=cfg.modeling.evaluation.topn_runs_included,
)
# Extract the top run
top_run_name, top_run_id = next(iter(selected_runs.items()))
logging.info(f"Selected top run for perf and bias: {top_run_name} - {top_run_id}")

# COMMAND ----------

# Update config with run and experiment ids
modeling.utils.update_run_metadata_in_toml(
    config_path="./config.toml",
    run_id=top_run_id,
    experiment_id=experiment_id,
)
