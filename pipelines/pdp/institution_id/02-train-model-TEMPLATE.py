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

import logging

import mlflow
import sklearn.metrics
from databricks.connect import DatabricksSession

from student_success_tool import configs, dataio, modeling, utils

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# Get job run id for automl run
job_run_id = utils.databricks.get_db_widget_param("job_run_id", default="interactive")

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

# project configuration should be stored in a config file in TOML format
cfg = dataio.read_config("./config-TEMPLATE.toml", schema=configs.pdp.PDPProjectConfig)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # read preprocessed dataset

# COMMAND ----------

df = dataio.read.from_delta_table(
    cfg.datasets.silver.preprocessed.table_path,
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
    df, cfg.datasets.silver.modeling.table_path, spark_session=spark
)

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
    "primary_metric": cfg.modeling.training.primary_metric,
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
    f"\n{training_params['primary_metric']} metric distribution = {summary.metric_distribution}"
)

dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
dbutils.jobs.taskValues.set(key="run_id", value=run_id)

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
# Adjust optimization metrics & topn_runs_included as needed
top_runs = modeling.evaluation.get_top_runs(
    experiment_id,
    optimization_metrics=[
        "test_recall_score",
        "val_recall_score",
        "test_roc_auc",
        "val_roc_auc",
        "test_log_loss",
        "val_log_loss",
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
            " and bias assessment" if evaluate_model_bias else "",
        )
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        df_pred = df.assign(
            **{
                cfg.pred_col: model.predict(df_features),
                cfg.pred_prob_col: modeling.inference.predict_probs(
                    df_features,
                    model,
                    feature_names=list(df_features.columns),
                    pos_label=cfg.pos_label,
                ),
            }
        )
        model_comp_fig = modeling.evaluation.plot_trained_models_comparison(
            experiment_id, cfg.modeling.training.primary_metric
        )
        modeling.evaluation.evaluate_performance(
            df_pred,
            target_col=cfg.target_col,
            pos_label=cfg.pos_label,
        )
        if evaluate_model_bias:
            modeling.bias_detection.evaluate_bias(
                df_pred,
                student_group_cols=cfg.student_group_cols,
                target_col=cfg.target_col,
                pos_label=cfg.pos_label,
            )
        logging.info("Run %s: Completed", run_id)
mlflow.end_run()

# COMMAND ----------

# Optional: Evaluate permutation importance for top model
# NOTE: This can be used for model diagnostics. It is NOT used
# in our standard evaluation process and not pulled into model cards.
model = mlflow.sklearn.load_model(f"runs:/{top_run_ids[0]}/model")
result = modeling.evaluation.compute_feature_permutation_importance(
    model,
    df_features,
    df[cfg.target_col],
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.log_loss, greater_is_better=False
    ),
    sample_weight=df[cfg.sample_weight_col],
    random_state=cfg.random_state,
)
ax = modeling.evaluation.plot_features_permutation_importance(
    result, feature_cols=df_features.columns
)
fig = ax.get_figure()
fig.tight_layout()
# save plot via mlflow into experiment artifacts folder
with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_figure(fig, "test_features_permutation_importance.png")
