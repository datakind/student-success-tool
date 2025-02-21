# Databricks notebook source
# MAGIC %md
# MAGIC # SST Train and Evaluate Model: exuni1
# MAGIC
# MAGIC Third step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: load a prepared modeling dataset, configure experiment tracking framework, then train and evaluate a predictive model.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Classification with AutoML](https://docs.databricks.com/en/machine-learning/automl/classification.html)
# MAGIC - [Databricks AutoML Python API reference](https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [exuni1 website](https://www.exuni1.edu)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install (minimal!) extra dependencies not provided by databricks runtime
# %pip install "student-success-tool==0.1.0"
# %pip install git+https://github.com/datakind/student-success-tool.git@develop --no-deps
# %pip install pandera

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
from student_success_tool import configs, modeling
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

# TODO: figure out what this is for
run_parameters = dict(dbutils.notebook.entry_point.getCurrentBindings())
job_run_id = run_parameters.get("job_run_id", "interactive")

# COMMAND ----------

config = configs.load_config("./config.toml", schema=configs.PDPProjectConfig)
config

# COMMAND ----------

training_params = {
    "job_run_id": job_run_id,
    "institution_id": config.institution_name,
    "student_id_col": config.train_evaluate_model.student_id_col,
    "target_col": config.train_evaluate_model.target_col,
    "split_col": config.train_evaluate_model.split_col,
    "sample_weight_col": config.train_evaluate_model.sample_weight_col,
    "pos_label": config.train_evaluate_model.pos_label,
    "optimization_metric": config.train_evaluate_model.primary_metric,
    "timeout_minutes": config.train_evaluate_model.timeout_minutes,
    "exclude_frameworks": config.train_evaluate_model.exclude_frameworks,
    "exclude_cols": sorted(
        set(
            (config.train_evaluate_model.exclude_cols or [])
            + (config.train_evaluate_model.student_group_cols or [])
        )
    ),
}
training_params

# COMMAND ----------

# MAGIC %md
# MAGIC # read modeling dataset

# COMMAND ----------

df = pdp.schemas.PDPLabeledDataSchema(
    pdp.dataio.read_data_from_delta_table(
        config.train_evaluate_model.dataset_table_path,
        spark_session=spark_session,
    )
)
print(f"rows x cols = {df.shape}")
df.head()

# COMMAND ----------

df[config.train_evaluate_model.target_col].value_counts(normalize=True)

# COMMAND ----------

if split_col := config.train_evaluate_model.split_col:
    print(df[split_col].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC # train model

# COMMAND ----------

# HACK
training_params["timeout_minutes"] = 5

# COMMAND ----------

summary = modeling.training.run_automl_classification(df, **training_params)

experiment_id = summary.experiment.experiment_id
experiment_run_id = summary.best_trial.mlflow_run_id
print(
    f"experiment_id: {experiment_id}"
    f"\n{training_params['optimization_metric']} metric distribution = {summary.metric_distribution}"
    f"\nbest trial experiment_run_id: {experiment_run_id}"
)

dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
dbutils.jobs.taskValues.set(key="experiment_run_id", value=experiment_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # evaluate model

# COMMAND ----------

pred_col = "pred"
pred_prob_col = "pred_prob"
# AutoML geneerates a split column if not manually specified
split_col = training_params.get("split_col", "_automl_split_col_0000")

# COMMAND ----------

model = summary.best_trial.load_model()
model

# COMMAND ----------

if evaluate_model_bias := (training_params.get("split_col") is not None):
    non_feature_cols = [
        training_params["student_id_col"],
        training_params["target_col"],
        split_col,
    ] + training_params["exclude_cols"]
    df_features = df.drop(columns=non_feature_cols)
#     train_df[prediction_col] = model.predict(features)
#     train_df[risk_score_col] = model.predict_proba(features)[:, 1]
# else:
#     train_df = extract_training_data_from_model(experiment_id)
#     train_df[prediction_col] = model.predict(train_df)
#     train_df[risk_score_col] = model.predict_proba(train_df)[:, 1]

# COMMAND ----------

df_pred = df.assign(
    **{
        pred_col: model.predict(df_features),
        pred_prob_col: model.predict_proba(df_features)[
            :, 1
        ],  # NOTE: assumes pos_label=True ?
    }
)
print(f"rows x cols = {df_pred.shape}")
df_pred.head()

# COMMAND ----------

pos_label = training_params["pos_label"]

target_col = training_params["target_col"]
pred_col = "pred"
pred_prob_col = "pred_prob"
# AutoML geneerates a split column if not manually specified
split_col = training_params.get("split_col", "_automl_split_col_0000")

calibration_dir = "calibration"
preds_dir = "preds"
sensitivity_dir = "sensitivity"

# COMMAND ----------

with mlflow.start_run(run_id=experiment_run_id) as run:
    model_comp_fig = modeling.evaluation.compare_trained_models_plot(
        experiment_id, training_params["optimization_metric"]
    )
    mlflow.log_figure(model_comp_fig, "primary_metric_by_model_type.png")

    for split_name, split_data in df_pred.groupby(split_col):
        tmp_path = f"/tmp/{split_name}_preds.csv"
        split_data.to_csv(tmp_path, header=True, index=False)
        mlflow.log_artifact(local_path=tmp_path, artifact_path=preds_dir)

        hist_fig, cal_fig, sla_fig = modeling.evaluation.create_evaluation_plots(
            split_data, pred_prob_col, target_col, pos_label, split_name
        )
        mlflow.log_figure(
            hist_fig,
            os.path.join(preds_dir, f"{split_name}_pred_probs_hist.png"),
        )
        mlflow.log_figure(
            cal_fig,
            os.path.join(calibration_dir, f"{split_name}_calibration_curve.png"),
        )
        mlflow.log_figure(
            sla_fig,
            os.path.join(sensitivity_dir, f"{split_name}_sla_at_{1.0:1.2f}pct.png"),
        )  # TODO: 0.01 as param?

        if not evaluate_model_bias:
            continue

        for group in config.train_evaluate_model.student_group_cols:
            cal_subgroup_fig, sla_subgroup_fig = (
                modeling.evaluation.create_evaluation_plots_by_subgroup(
                    split_data, pred_prob_col, target_col, pos_label, group, split_name
                )
            )
            mlflow.log_figure(
                cal_subgroup_fig,
                os.path.join(
                    calibration_dir, f"{split_name}_{group}_calibration_curve.png"
                ),
            )
            mlflow.log_figure(
                sla_subgroup_fig,
                os.path.join(
                    sensitivity_dir,
                    f"{split_name}_{group}_sla_at_{1.0:1.2f}pct.png",
                ),
            )

            group_metrics = []
            for subgroup, subgroup_data in split_data.groupby(group):
                labels = subgroup_data[target_col]
                preds = subgroup_data[pred_col]
                pred_probs = subgroup_data[pred_prob_col]

                subgroup_metrics = {
                    "subgroup": subgroup,
                    "num": len(subgroup_data),
                    "actual_target_prevalence": labels.mean(),
                    "pred_target_prevalence": preds.mean(),
                    "accuracy": sklearn.metrics.accuracy_score(labels, preds),
                    "precision": sklearn.metrics.precision_score(
                        labels, preds, pos_label=pos_label, zero_division=np.nan
                    ),
                    "recall": sklearn.metrics.recall_score(
                        labels, preds, pos_label=pos_label, zero_division=np.nan
                    ),
                    "log_loss": sklearn.metrics.log_loss(
                        labels, preds, labels=[False, True]
                    ),
                }
                group_metrics.append(subgroup_metrics)
                # log metrics
                for metric, value in subgroup_metrics.items():
                    if metric not in {"subgroup", "num", "actual_target_prevalence"}:
                        mlflow.log_metric(
                            f"{split_name}_{group}_metrics/{metric}_subgroup{subgroup}",
                            value,
                        )

            # also store metrics in a table for easier comparison
            df_group_metrics = pd.DataFrame(group_metrics)
            metrics_tmp_path = f"/tmp/{split_name}_{group}_metrics.csv"
            df_group_metrics.to_csv(metrics_tmp_path, index=False)
            mlflow.log_artifact(
                local_path=metrics_tmp_path, artifact_path="subgroup_metrics"
            )
            print(df_group_metrics)
