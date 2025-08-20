# Databricks notebook source
# MAGIC %md
# MAGIC # SST Make and Explain Predictions: [SCHOOL]
# MAGIC
# MAGIC Fourth step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: generate predictions and feature importances for new (unlabeled) data.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [SCHOOL WEBSITE](https://example.com)

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

# MAGIC %restart_python

# COMMAND ----------

import functools as ft
import logging

import mlflow
import pandas as pd
import shap
from databricks.connect import DatabricksSession

from student_success_tool import configs, dataio, modeling
from student_success_tool.modeling import h2o_modeling

import h2o

h2o.init()
h2o.display.toggle_user_tips(False)

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger
logging.getLogger("h2o").setLevel(logging.WARN) # ignore h2o logger since it gets verbose

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# Databricks logs every instance that uses sklearn or other modelling libraries
# to MLFlow experiments... which we don't want
mlflow.autolog(disable=True)

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

# Load human-friendly PDP feature names
features_table = dataio.read_features_table("./features_table.toml")

# COMMAND ----------

# MAGIC %md
# MAGIC # load artifacts

# COMMAND ----------

df = dataio.read.from_delta_table(
    cfg.datasets.silver["modeling"].table_path,
    spark_session=spark,
)
df.head()

# COMMAND ----------

model = h2o_modeling.utils.load_h2o_model(
    cfg.model.run_id
)
model

# COMMAND ----------

model_feature_names = h2o_modeling.inference.get_h2o_used_features(model)
logging.info(
    "model uses %s features: %s", len(model_feature_names), model_feature_names
)

# COMMAND ----------

df_train = h2o_modeling.evaluation.extract_training_data_from_model(cfg.model.experiment_id)
if cfg.split_col:
    df_train = df_train.loc[df_train[cfg.split_col].eq("train"), :]
df_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # preprocess data

# COMMAND ----------

if cfg.split_col and cfg.split_col in df.columns:
    df_test = df.loc[df[cfg.split_col].eq("test"), :]
else:
    df_test = df.copy(deep=True)

student_ids = df_test.student_id

# Load and transform using sklearn imputer
df_test = h2o_modeling.imputation.SklearnImputerWrapper.load_and_transform(
    df_test,
    run_id=cfg.model.run_id
)
df_test['student_id'] = student_ids

# COMMAND ----------

# MAGIC %md
# MAGIC # make predictions

# COMMAND ----------

features = df_test.loc[:, model_feature_names]
unique_ids = df_test[cfg.student_id_col]

# COMMAND ----------

pred_probs = h2o_modeling.inference.predict_probs_h2o(
    features,
    model=model,
    feature_names=model_feature_names,
    pos_label=cfg.pos_label,
)
print(pred_probs.shape)
pd.Series(pred_probs).describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # explain predictions

# COMMAND ----------

train = h2o.H2OFrame(df_train)
h2o_features = h2o.H2OFrame(features)

contribs_df, preprocessed_df = h2o_modeling.inference.compute_h2o_shap_contributions(
    model=model,
    h2o_frame=h2o_features,
    background_data=train,
)
contribs_df

# COMMAND ----------

# Group one-hot encoding and missing value flags
grouped_contribs_df = h2o_modeling.inference.group_shap_values(contribs_df, group_missing_flags=True)
grouped_features = h2o_modeling.inference.group_feature_values(features, group_missing_flags=True)

# COMMAND ----------

with mlflow.start_run(run_id=cfg.model.run_id):
    # Create & log SHAP summary plot (default to group missing flags)
    h2o_modeling.inference.plot_grouped_shap(
        contribs_df=contribs_df,
        preprocessed_df=preprocessed_df,
        original_df=features,
        group_missing_flags=True,
    )

    # Create & log ranked features by SHAP magnitude
    selected_features_df = modeling.inference.generate_ranked_feature_table(
        grouped_features,
        grouped_contribs_df.to_numpy(),
        features_table=features_table,
    )

selected_features_df

# COMMAND ----------

# MAGIC %md
# MAGIC # finalize results

# COMMAND ----------

# Provide output using top features, SHAP values, and support scores
result = modeling.inference.select_top_features_for_display(
    grouped_features,
    unique_ids,
    pred_probs,
    grouped_contribs_df.to_numpy(),
    n_features=10,
    features_table=features_table,
    needs_support_threshold_prob=cfg.inference.min_prob_pos_label,
)
result

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    result, cfg.datasets.gold.advisor_output.table_path, spark_session=spark
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Front-End Tables

# COMMAND ----------

# Log MLFlow confusion matrix & roc table figures in silver schema

with mlflow.start_run() as run:
    confusion_matrix = modeling.evaluation.log_confusion_matrix(
        institution_id=cfg.institution_id,
        automl_run_id=cfg.model.run_id,
    )

    # Log roc curve table for front-end
    roc_logs = modeling.evaluation.log_roc_table(
        institution_id=cfg.institution_id,
        automl_run_id=cfg.model.run_id,
        modeling_dataset_name=cfg.datasets.silver.modeling.table_path,
    )

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    shap_feature_importance,
    f"staging_sst_01.{cfg.institution_id}_silver.training_{cfg.model.run_id}_shap_feature_importance",
    spark_session=spark,
)

# COMMAND ----------

support_score_distribution = modeling.inference.support_score_distribution_table(
    df_serving=grouped_features,
    unique_ids=unique_ids,
    pred_probs=pred_probs,
    shap_values=grouped_contribs_df.to_numpy(),
    inference_params=cfg.inference.dict(),
)
support_score_distribution

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    support_score_distribution,
    f"staging_sst_01.{cfg.institution_id}_silver.training_{cfg.model.run_id}_support_overview",
    spark_session=spark,
)
