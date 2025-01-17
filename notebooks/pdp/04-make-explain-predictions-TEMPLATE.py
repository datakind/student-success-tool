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

# install dependencies, most of which should come through our 1st-party SST package
# %pip install "student-success-tool==0.1.0" --no-deps
# %pip install git+https://github.com/datakind/student-success-tool.git@develop --no-deps
# %pip install pandera

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import functools as ft
import logging
import typing as t

import mlflow
import numpy as np
import pandas as pd
import shap
import sklearn.inspection
import sklearn.metrics
from databricks.connect import DatabricksSession

# from databricks.sdk.runtime import dbutils
# from py4j.protocol import Py4JJavaError
# from pyspark import SparkContext
# from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from student_success_tool.analysis.pdp import dataio
from student_success_tool.modeling import inference, utils

# COMMAND ----------

logging.getLogger("root").setLevel(logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark_session = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# Databricks logs every instance that uses sklearn or other modelling libraries
# to MLFlow experiments... which we don't want
mlflow.autolog(disable=True)
mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## configuration

# COMMAND ----------

# TODO TODO TODO: use project config

train_sample_size = 100
validate_sample_size = 100

institution_id = "INSTITUTION_ID"
best_model_run_id = "BEST_MODEL_RUN_ID"
student_id_col = "student_guid"
target_col = "target"
split_col = "split"
sample_weight_col = "sample_weight"
pos_label = True

model_type = "sklearn"
labeled_data_path = "CATALOG.SCHEMA.TABLE_NAME"  # "TODO"
unlabeled_data_path = None

# COMMAND ----------

# MAGIC %md
# MAGIC # load model and data

# COMMAND ----------


# TODO: move this into sst package
def mlflow_load_model(model_uri: str, model_type: str):
    load_model_func = (
        mlflow.sklearn.load_model
        if model_type == "sklearn"
        else mlflow.xgboost.load_model
        if model_type == "xgboost"
        else mlflow.lightgbm.load_model
        if model_type == "lightgbm"
        else mlflow.pyfunc.load_model
    )
    model = load_model_func(f"runs:/{best_model_run_id}/model")
    logging.info("mlflow '%s' model loaded from '%s'", model_type, model_uri)
    return model


def predict_proba(
    df: pd.DataFrame, *, model, pos_label: bool | str = True
) -> pd.Series:
    return pd.Series(
        model.predict_proba(df)[:, model.classes_.tolist().index(pos_label)]
    )


# COMMAND ----------

model = mlflow_load_model(f"runs:/{best_model_run_id}/model", model_type)
model_features = model.named_steps["column_selector"].get_params()["cols"]
logging.info(
    "model uses %s features: %s", len(model_features), ", ".join(model_features)
)

# COMMAND ----------

model_features = model.named_steps["column_selector"].get_params()["cols"]
print(len(model_features))

# COMMAND ----------

df_labeled = dataio.read_data_from_delta_table(
    labeled_data_path, spark_session=spark_session
)
print(df_labeled.shape)
df_labeled.head()

# COMMAND ----------

if unlabeled_data_path:
    df_unlabeled = dataio.read_data_from_delta_table(
        unlabeled_data_path, spark_session=spark_session
    )
else:
    df_unlabeled = df_labeled.loc[df_labeled[split_col].eq("test"), :].drop(
        columns=target_col
    )
print(df_unlabeled.shape)
df_unlabeled.head()

# COMMAND ----------

pred_probs = predict_proba(df_unlabeled, model=model)
pred_probs.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC # initialize SHAP explainer

# COMMAND ----------

df_train = df_labeled.loc[df_labeled[split_col].eq("train"), :]
# SHAP can't explain models using data with nulls
# so, impute nulls using the mode (most frequent values)
mode = df_train.mode().iloc[0]
# sample background data for SHAP Explainer
train_sample = (
    df_train.sample(n=min(train_sample_size, df_train.shape[0]), random_state=1)
    .fillna(mode)
    .loc[:, model_features]
)
train_sample

# COMMAND ----------


def predict_proba_v3(
    X,
    *,
    model,
    col_names: t.Optional[list[str]] = None,
    pos_label: t.Optional[bool | str] = None,
) -> np.ndarray:
    if col_names is None:
        col_names = model.named_steps["column_selector"].get_params()["cols"]
    pred_probs = model.predict_proba(pd.DataFrame(data=X, columns=col_names))
    if pos_label is not None:
        return pred_probs[:, model.classes_.tolist().index(pos_label)]
    else:
        return pred_probs


def predict_proba_v2(X, *, model, pos_label: bool | str = True):
    model_features = model.named_steps["column_selector"].get_params()["cols"]
    pred_probs = model.predict_proba(pd.DataFrame(data=X, columns=model_features))
    return pred_probs[:, model.classes_.tolist().index(pos_label)]


# COMMAND ----------

# import shap
# import sklearn

# X, y = shap.datasets.adult()
# m = sklearn.linear_model.LogisticRegression().fit(X, y)
# explainer = shap.explainers.Permutation(m.predict_proba, X)
# shap_values = explainer(X[:100])
# shap.plots.bar(shap_values[..., 1])

# COMMAND ----------

# explainer = shap.explainers.Permutation(model.predict_proba, train_sample)
# explainer = shap.explainers.KernelExplainer(model.predict_proba, train_sample)
explainer = shap.explainers.KernelExplainer(
    ft.partial(
        predict_proba_v3, model=model, col_names=model_features, pos_label=pos_label
    ),
    train_sample,
    link="identity",
)
explainer

# COMMAND ----------

shap_schema = StructType(
    [StructField(student_id_col, StringType(), nullable=False)]
    + [StructField(col, FloatType(), nullable=False) for col in model_features]
)

df_shap_values = (
    spark.createDataFrame(df_unlabeled.drop(columns=[split_col, sample_weight_col]))  # noqa: F821
    .repartition(sc.defaultParallelism)  # noqa: F821
    .mapInPandas(
        ft.partial(
            inference.calculate_shap_values_spark_udf,
            student_id_col=student_id_col,
            model_features=model_features,
            explainer=explainer,
            mode=mode,
        ),
        schema=shap_schema,
    )
    .toPandas()
    .set_index(student_id_col)
    .reindex(df_unlabeled[student_id_col])
    .reset_index(drop=False)
)
df_shap_values

# COMMAND ----------

shap.summary_plot(
    df_shap_values[model_features].to_numpy(),
    df_unlabeled[model_features],
    class_names=model.classes_,
    # show=False, ???
)

# COMMAND ----------

features_table = utils.load_features_table("assets/pdp/features_table.toml")
result = inference.select_top_features_for_display(
    df_unlabeled.loc[:, model_features],
    df_unlabeled[student_id_col],
    pred_probs,
    df_shap_values[model_features].to_numpy(),
    n_features=5,
    features_table=features_table,
    needs_support_threshold_prob=0.5,
)
result

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO:
# MAGIC
# MAGIC - save plots and results in a nice form in a place that makes sense

# COMMAND ----------

# MAGIC %md
# MAGIC ## haxx

# COMMAND ----------


result = sklearn.inspection.permutation_importance(
    model,
    train_sample.drop(columns=target_col),
    train_sample[target_col],
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.log_loss, greater_is_better=False
    ),
    n_repeats=10,
)

# COMMAND ----------

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=train_sample.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10, figsize=(10, 10))
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in accuracy score")
ax.figure.tight_layout()

# COMMAND ----------
