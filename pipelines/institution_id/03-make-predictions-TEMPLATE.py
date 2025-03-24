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

# WARNING: AutoML/mlflow expect particular packages within certain version constraints
# overriding existing installs can result in errors and inability to load trained models
# %pip install "student-success-tool==0.1.1" --no-deps
# %pip install "git+https://github.com/datakind/student-success-tool.git@develop" --no-deps

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import functools as ft
import logging
import typing as t

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from py4j.protocol import Py4JJavaError
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from student_success_tool import dataio, modeling, schemas
from student_success_tool.modeling import inference

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

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

# check if we're running this notebook as a "job" in a "workflow"
# if not, assume this is a prediction workflow
try:
    run_type = dbutils.widgets.get("run_type")
    dataset_name = dbutils.widgets.get("dataset_name")
    model_name = dbutils.widgets.get("model_name")
except Py4JJavaError:
    run_type = "predict"
    # TODO: specify dataset and model name, as (to be) included in project config
    dataset_name = "DATASET_NAME"
    model_name = "MODEL_NAME"

logging.info(
    "'%s' run of notebook using '%s' dataset w/ '%s' model",
    run_type,
    dataset_name,
    model_name,
)
# TODO: do we need this?
# if run_type != "train":
#     logging.warning(
#         "this notebook is meant for model training; "
#         "should you be running it in '%s' mode?",
#         run_type,
#     )

# COMMAND ----------

# MAGIC %md
# MAGIC ## import school-specific code

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
# MAGIC # load artifacts

# COMMAND ----------

df = schemas.pdp.PDPLabeledDataSchema(
    dataio.read.from_delta_table(
        cfg.datasets[dataset_name].preprocessed.table_path,
        spark_session=spark,
    )
)
df.head()

# COMMAND ----------


# TODO: get this functionality into public repo's modeling.inference?
def predict_proba(
    X,
    model,
    *,
    feature_names: t.Optional[list[str]] = None,
    pos_label: t.Optional[bool | str] = None,
) -> np.ndarray:
    """ """
    if feature_names is None:
        feature_names = model.named_steps["column_selector"].get_params()["cols"]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(data=X, columns=feature_names)
    else:
        assert X.shape[1] == len(feature_names)
    pred_probs = model.predict_proba(X)
    if pos_label is not None:
        return pred_probs[:, model.classes_.tolist().index(pos_label)]
    else:
        return pred_probs


# COMMAND ----------

model = modeling.utils.load_mlflow_model(
    cfg.models[model_name].mlflow_model_uri,
    cfg.models[model_name].framework,
)
model

# COMMAND ----------

model_feature_names = model.named_steps["column_selector"].get_params()["cols"]
logging.info(
    "model uses %s features: %s", len(model_feature_names), model_feature_names
)

# COMMAND ----------

features_table = dataio.read_features_table("assets/pdp/features_table.toml")

# COMMAND ----------

df_train = modeling.evaluation.extract_training_data_from_model(
    cfg.models[model_name].experiment_id
)
if cfg.split_col:
    df_train = df_train.loc[df_train[cfg.split_col].eq("train"), :]
df_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # make predictions

# COMMAND ----------

# TODO: load inference params from the project config
# okay to hard-code it first then add it to the config later
# inference_params = cfg.inference.model_dump()
inference_params = {
    "num_top_features": 5,
    "min_prob_pos_label": 0.5,
}

# COMMAND ----------

if cfg.split_col and cfg.split_col in df.columns:
    df_test = df.loc[df[cfg.split_col].eq("test"), :]
else:
    df_test = df.copy(deep=True)

if cfg.target_col in df.columns:
    df_test = df_test.drop(columns=cfg.target_col)
    # TODO: do we want to sample this dataset for labeled data?
    df_test = df_test.sample(n=200)

# COMMAND ----------

features = df_test.loc[:, model_feature_names]
unique_ids = df_test[cfg.student_id_col]

# COMMAND ----------

pred_probs = predict_proba(
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

shap_ref_data_size = 100
# SHAP can't explain models using data with nulls
# so, impute nulls using the mode (most frequent values)
train_mode = df_train.mode().iloc[0]
# sample training dataset as "reference" data for SHAP Explainer
df_ref = (
    df_train.sample(
        n=min(shap_ref_data_size, df_train.shape[0]),
        random_state=cfg.random_state,
    )
    .fillna(train_mode)
    .loc[:, model_feature_names]
)
df_ref.shape

# COMMAND ----------

explainer = shap.explainers.KernelExplainer(
    ft.partial(
        predict_proba,
        model=model,
        feature_names=model_feature_names,
        pos_label=cfg.pos_label,
    ),
    df_ref,
    link="identity",
)
explainer

# COMMAND ----------

shap_schema = StructType(
    [StructField(cfg.student_id_col, StringType(), nullable=False)]
    + [StructField(col, FloatType(), nullable=False) for col in model_feature_names]
)

df_shap_values = (
    spark.createDataFrame(
        df_test.reindex(columns=model_feature_names + [cfg.student_id_col])
    )  # noqa: F821
    .repartition(sc.defaultParallelism)  # noqa: F821
    .mapInPandas(
        ft.partial(
            inference.calculate_shap_values_spark_udf,
            student_id_col=cfg.student_id_col,
            model_features=model_feature_names,
            explainer=explainer,
            mode=train_mode,
        ),
        schema=shap_schema,
    )
    .toPandas()
    .set_index(cfg.student_id_col)
    .reindex(df_test[cfg.student_id_col])
    .reset_index(drop=False)
)
df_shap_values

# COMMAND ----------

shap.summary_plot(
    df_shap_values.loc[:, model_feature_names].to_numpy(),
    df_test.loc[:, model_feature_names],
    class_names=model.classes_,
    max_display=20,
    show=False,
)
shap_fig = plt.gcf()
# save shap summary plot via mlflow into experiment artifacts folder
with mlflow.start_run(run_id=cfg.models[model_name].run_id) as run:
    mlflow.log_figure(
        shap_fig, f"shap_summary_{dataset_name}_dataset_{df_ref.shape[0]}_ref_rows.png"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # finalize results

# COMMAND ----------

features_table = dataio.read_features_table("assets/pdp/features_table.toml")
result = inference.select_top_features_for_display(
    features,
    unique_ids,
    pred_probs,
    df_shap_values[model_feature_names].to_numpy(),
    n_features=inference_params["num_top_features"],
    features_table=features_table,
    needs_support_threshold_prob=inference_params["min_prob_pos_label"],
)
result

# COMMAND ----------

# MAGIC %md
# MAGIC # TODO:
# MAGIC
# MAGIC - how / where to save final results?
# MAGIC - do we want to save predictions separately / additionally from the "display" format?

# COMMAND ----------
