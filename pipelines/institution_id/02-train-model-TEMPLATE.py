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

# WARNING: AutoML/mlflow expect particular packages within certain version constraints
# overriding existing installs can result in errors and inability to load trained models
# install (minimal!) extra dependencies not provided by databricks runtime
# %pip install "student-success-tool==0.1.1" --no-deps
# %pip install git+https://github.com/datakind/student-success-tool.git@develop --no-deps

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import os

import mlflow
import numpy as np
import pandas as pd
import sklearn.metrics
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils

from student_success_tool import dataio, modeling, schemas, utils

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

run_type = utils.databricks.get_db_widget_param("run_type", default="train")
dataset_name = utils.databricks.get_db_widget_param("dataset_name", default="labeled")
# should this be "job_id" and "run_id", separately
job_run_id = utils.databricks.get_db_widget_param("job_run_id", default="interactive")

logging.info(
    "'%s' run (id=%s) of notebook using dataset_name=%s",
    run_type,
    job_run_id,
    dataset_name,
)

# COMMAND ----------

assert run_type == "train"

# COMMAND ----------

# project configuration should be stored in a config file in TOML format
# it'll start out with just basic info: institution_id, institution_name
# but as each step of the pipeline gets built, more parameters will be moved
# from hard-coded notebook variables to shareable, persistent config fields
cfg = dataio.read_config("./config-TEMPLATE.toml", schema=schemas.pdp.PDPProjectConfig)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # read modeling dataset

# COMMAND ----------

df = dataio.read.from_delta_table(
    cfg.datasets[dataset_name].preprocessed.table_path,
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

# TODO: load feature selection params from the project config
# okay to hard-code it first then add it to the config later
selection_params = cfg.modeling.feature_selection.model_dump()
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

# MAGIC %md
# MAGIC # train model

# COMMAND ----------

# re-enable mlflow's autologging
mlflow.autolog(disable=False)

# COMMAND ----------

training_params = {
    "job_run_id": job_run_id,
    "institution_id": cfg.institution_id,
    "student_id_col": cfg.student_id_col,
    "target_col": cfg.target_col,
    "split_col": cfg.split_col,
    "sample_weight_col": cfg.sample_weight_col,
    "pos_label": cfg.pos_label,
    "optimization_metric": cfg.modeling.training.primary_metric,
    "timeout_minutes": cfg.modeling.training.timeout_minutes,
    "exclude_frameworks": cfg.modeling.training.exclude_frameworks,
    "exclude_cols": sorted(
        set((cfg.modeling.training.exclude_cols or []) + (cfg.student_group_cols or []))
    ),
}
logging.info("training params = %s", training_params)

# COMMAND ----------

summary = modeling.training.run_automl_classification(df, **training_params)

experiment_id = summary.experiment.experiment_id
run_id = summary.best_trial.mlflow_run_id
print(
    f"experiment_id: {experiment_id}"
    f"\nbest trial run_id: {run_id}"
    f"\n{training_params['optimization_metric']} metric distribution = {summary.metric_distribution}"
)

dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
dbutils.jobs.taskValues.set(key="run_id", value=run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # evaluate model

# COMMAND ----------

# HACK: Evaluate an experiment you've already trained
# experiment_id = cfg.models["graduation"].experiment_id

calibration_dir = "calibration"
preds_dir = "preds"
sensitivity_dir = "sensitivity"

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

evaluation_params = {
    "student_group_cols": cfg.student_group_cols,
    "target_col": training_params.get("target_col"),
    "optimization_metric": training_params.get("optimization_metric"),
    "pos_label": training_params.get("pos_label"),
    "pred_col": cfg.pred_col,
    "pred_prob_col": cfg.pred_prob_col,
    "experiment_id": experiment_id,
    "top_run_ids": cfg.modeling.evaluation.topn_runs_included,
}
logging.info("evaluation params = %s", evaluation_params)

# COMMAND ----------

# Get top runs from experiment for evaluation
top_run_ids = modeling.evaluation.get_top_run_ids(**evaluation_params)

# Iterate through runs and evaluate performance & bias
for run_id in top_run_ids:
    with mlflow.start_run(run_id=run_id) as run:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        df_pred = df.assign(
            **{
                evaluation_params.get("pred_col"): model.predict(df_features),
                evaluation_params.get("pred_prob_col"): model.predict_proba(df_features)[:, 1],
            }
        )
        logging.info(f"Processing run {run_id} - rows x cols = {df_pred.shape}")
        model_comp_fig = modeling.evaluation.compare_trained_models_plot(
            evaluation_params.get("experiment_id"), evaluation_params.get("optimization_metric")
        )
        mlflow.log_figure(model_comp_fig, "model_comparison.png")

        all_model_flags = [] 

        for split_name, split_data in df_pred.groupby(split_col):
            split_data.to_csv(f"/tmp/{split_name}_preds.csv", header=True, index=False)
            mlflow.log_artifact(
                local_path=f"/tmp/{split_name}_preds.csv",
                artifact_path=preds_dir,
            )

            hist_fig, cal_fig, sla_fig = modeling.evaluation.create_evaluation_plots(
                split_data,
                split_name,
                **evaluation_params,
            )
            mlflow.log_figure(
                hist_fig,
                os.path.join(preds_dir, f"{split_name}_hist.png"),
            )
            mlflow.log_figure(
                cal_fig,
                os.path.join(calibration_dir, f"{split_name}_calibration.png"),
            )
            mlflow.log_figure(
                sla_fig,
                os.path.join(sensitivity_dir, f"{split_name}_sla.png"),
            )
            
            if evaluate_model_bias:
                split_flags = modeling.bias_detection.evaluate_bias(
                    run_id,
                    df_model_flags,
                    split_data,
                    split_name,
                    **evaluation_params
                )
                all_model_flags.extend(split_flags)
        
        log_bias_flags_to_mlflow(all_model_flags)

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC # evaluate model

# COMMAND ----------

# Evaluation permutation importance for top model
model = mlflow.sklearn.load_model(f"runs:/{top_run_ids[0]}/model")
ax = modeling.evaluation.plot_features_permutation_importance(
    model,
    df_features,
    df[cfg.target_col],
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.log_loss, greater_is_better=False
    ),
    sample_weight=df[cfg.sample_weight_col],
    random_state=cfg.random_state,
)
fig = ax.get_figure()
fig.tight_layout()
# save plot via mlflow into experiment artifacts folder
with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_figure(fig, "test_features_permutation_importance.png")
