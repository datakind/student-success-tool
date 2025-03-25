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
# %pip install "student-success-tool==0.1.0" --no-deps
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
import sklearn.inspection
import sklearn.metrics
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from py4j.protocol import Py4JJavaError

from student_success_tool import dataio, modeling, schemas

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

# TODO(Burton?): figure out if/how this should be used
run_parameters = dict(dbutils.notebook.entry_point.getCurrentBindings())

# check if we're running this notebook as a "job" in a "workflow"
# if not, assume this is a training workflow using labeled data
try:
    run_type = dbutils.widgets.get("run_type")
    dataset_name = dbutils.widgets.get("dataset_name")
except Py4JJavaError:
    run_type = "train"
    dataset_name = "labeled"

logging.info("'%s' run of notebook using '%s' dataset", run_type, dataset_name)
# just in case!
if run_type != "train":
    logging.warning(
        "this notebook is meant for model training; "
        "should you be running it in '%s' mode?",
        run_type,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## import school-specific code

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# project configuration should be stored in a config file in TOML format
# it'll start out with just basic info: institution_id, institution_name
# but as each step of the pipeline gets built, more parameters will be moved
# from hard-coded notebook variables to shareable, persistent config fields
cfg = dataio.read_config(
    "./config-TEMPLATE.toml", schema=schemas.pdp.PDPProjectConfigV2
)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # read modeling dataset

# COMMAND ----------

df = schemas.pdp.PDPLabeledDataSchema(
    dataio.read.from_delta_table(
        cfg.datasets[dataset_name].preprocessed.table_path,
        spark_session=spark,
    )
)
print(f"rows x cols = {df.shape}")
df.head()

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
try:
    selection_params = cfg.modeling.feature_selection.model_dump()
    logging.info("selection params = %s", selection_params)
except AttributeError:
    selection_params = {
        "non_feature_cols": cfg.non_feature_cols,
        "force_include_cols": [],
        "incomplete_threshold": 0.5,
        "low_variance_threshold": 0.0,
        "collinear_threshold": 10.0,
    }

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
    "job_run_id": run_parameters.get("job_run_id", "interactive"),
    "institution_id": cfg.institution_id,
    "student_id_col": cfg.student_id_col,
    "target_col": cfg.target_col,
    "split_col": cfg.split_col,
    "sample_weight_col": cfg.sample_weight_col,
    "pos_label": cfg.pos_label,
}
# TODO: load feature selection params from the project config
# okay to hard-code it first then add it to the config later
try:
    training_params |= cfg.modeling.training.model_dump()
except AttributeError:
    training_params |= {
        "optimization_metric": "log_loss",
        "timeout_minutes": 15,
        "exclude_frameworks": [],
        "exclude_cols": cfg.student_group_cols or [],
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

calibration_dir = "calibration"
preds_dir = "preds"
sensitivity_dir = "sensitivity"

# NOTE: AutoML geneerates a split column if not manually specified
split_col = training_params.get("split_col", "_automl_split_col_0000")

# COMMAND ----------

# only possible to do bias evaluation if you specify a split col for train/test/validate
# AutoML doesn't preserve student ids the training set, which we need for [reasons]
if evaluate_model_bias := (training_params.get("split_col") is not None):
    non_feature_cols = sorted(
        set(
            [
                training_params["student_id_col"],
                training_params["target_col"],
                split_col,
            ]
            + training_params["exclude_cols"]
        )
    )
    df_features = df.drop(columns=non_feature_cols)
else:
    df_features = modeling.evaluation.extract_training_data_from_model(experiment_id)

# COMMAND ----------

top_x = 3  # Change this to the number of top runs you want to evaluate
optimization_metric = training_params["optimization_metric"]

# Fetch top X runs
runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=[f"metrics.{optimization_metric} DESC"],  # Sort by metric, adjust order if lower is better
    max_results=top_x
)

# Extract run IDs
top_run_ids = runs["run_id"].tolist()

# COMMAND ----------

for run_id in top_run_ids:
    with mlflow.start_run(run_id=run_id) as run:
        print(f"Processing run {run_id} - rows x cols = {df_pred.shape}")

        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")  
        df_all_flags = pd.DataFrame() 
        df_pred = df.assign(
            **{
                cfg.pred_col: model.predict(df_features),
                cfg.pred_prob_col: model.predict_proba(df_features)[:, 1],
            }
        )
        model_comp_fig = modeling.evaluation.compare_trained_models_plot(experiment_id, optimization_metric)
        mlflow.log_figure(model_comp_fig, f"model_comparison_{run_id}.png")

        for split_name, split_data in df_pred.groupby(split_col):
            split_data.to_csv(f"/tmp/{split_name}_preds_{run_id}.csv", header=True, index=False)
            mlflow.log_artifact(local_path=f"/tmp/{split_name}_preds_{run_id}.csv", artifact_path=preds_dir)

            hist_fig, cal_fig, sla_fig = modeling.evaluation.create_evaluation_plots(
                split_data, cfg['pred_prob_col'], cfg['target_col'], cfg['pos_label'], split_name
            )
            mlflow.log_figure(hist_fig, os.path.join(preds_dir, f"{split_name}_hist_{run_id}.png"))
            mlflow.log_figure(cal_fig, os.path.join(calibration_dir, f"{split_name}_calibration_{run_id}.png"))
            mlflow.log_figure(sla_fig, os.path.join(sensitivity_dir, f"{split_name}_sla_{run_id}.png"))

            if evaluate_model_bias:
                for group in cfg['student_group_cols']:
                    group_metrics = []
                    fnpr_data = []
                    for subgroup, subgroup_data in split_data.groupby(group):
                        labels = subgroup_data[cfg['target_col']]
                        preds = subgroup_data[cfg['pred_col']]
                        pred_probs = subgroup_data[cfg['pred_prob_col']]
                        fnpr, fnpr_lower, fnpr_upper = modeling.bias_detection.calculate_fnpr_and_ci(labels, preds)

                        fnpr_data.append({
                            'group': group,
                            'subgroup': subgroup,
                            'fnpr': fnpr,
                            'ci': (fnpr_lower, fnpr_upper),
                            'size': len(subgroup_data)
                        })

                        subgroup_metrics = {
                            "Run ID": run_id,
                            "Subgroup": subgroup,
                            "Number of Samples": len(subgroup_data),
                            "Actual Target Prevalence": round(labels.mean(), 2),
                            "Predicted Target Prevalence": round(preds.mean(), 2),
                            "FNPR": round(fnpr, 2) if not pd.isna(fnpr) else "Insufficient # of FN or TP",
                            "Accuracy": round(sklearn.metrics.accuracy_score(labels, preds), 2),
                            "Precision": round(sklearn.metrics.precision_score(labels, preds, pos_label=cfg['pos_label'], zero_division=np.nan), 2),
                            "Recall": round(sklearn.metrics.recall_score(labels, preds, pos_label=cfg['pos_label'], zero_division=np.nan), 2),
                            "Log Loss": round(sklearn.metrics.log_loss(labels, pred_probs, labels=[False, True]), 2),
                        }

                        for metric, value in subgroup_metrics.items():
                            if metric not in {"Subgroup", "Number of Samples"} and not pd.isna(fnpr):
                                mlflow.log_metric(f"{split_name}_{group}_metrics/{metric}_subgroup{subgroup}_run{run_id}", value)

                        group_metrics.append(subgroup_metrics)

                    df_group_metrics = pd.DataFrame(group_metrics)
                    metrics_tmp_path = f"/tmp/{split_name}_{group}_metrics_run{run_id}.csv"
                    df_group_metrics.to_csv(metrics_tmp_path, index=False)
                    mlflow.log_artifact(local_path=metrics_tmp_path, artifact_path="subgroup_metrics")

                    bias_flags = modeling.bias_detection.flag_bias(fnpr_data, split_name)

                    for flag in bias_flags:
                        if flag['flag'] != "🟢 NO BIAS":
                            print(f"Run {run_id}: {flag['group']} on {flag['dataset']} - {flag['subgroups']}, FNPR Difference: {flag['difference']:.2f}% ({flag['type']}) [{flag['flag']}]")

                    df_bias_flags = pd.DataFrame(bias_flags)
                    df_all_flags = pd.concat([df_all_flags, df_bias_flags], ignore_index=True)

flag_names = {'🟢 NO BIAS': 'no_bias', '🟡 LOW BIAS': 'low_bias', '🟠 MODERATE BIAS': 'moderate_bias', '🔴 HIGH BIAS': 'high_bias'}
for flag in flag_names.keys():
    flag_name = flag_names[flag]
    df_flag = df_all_flags[df_all_flags['flag'] == flag].sort_values(by='difference', ascending=False)
    bias_tmp_path = f"/tmp/{flag_name}_flags.csv"
    df_flag.to_csv(bias_tmp_path, index=False)
    mlflow.log_artifact(local_path=bias_tmp_path, artifact_path="bias_flags")

# COMMAND ----------

# TODO(Burton?): clean this up and incorporate it into modeling.evaluation
result = sklearn.inspection.permutation_importance(
    model,
    df_features.drop(columns=cfg.target_col),
    df_features[cfg.target_col],
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.log_loss, greater_is_better=False
    ),
    n_repeats=5,
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=df_features.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10, figsize=(10, 10))
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()
