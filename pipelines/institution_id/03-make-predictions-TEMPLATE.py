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

# %pip install "student-success-tool==0.3.1"
# %pip install "pandas==1.5.3"
# %pip install "scikit-learn==1.3.0"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import functools as ft
import logging

import mlflow
import pandas as pd
import shap
from databricks.connect import DatabricksSession
from pyspark.sql.types import FloatType, StringType, StructField, StructType

from student_success_tool import configs, dataio, modeling
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

# MAGIC %md
# MAGIC ## import school-specific code

# COMMAND ----------

# project configuration should be stored in a config file in TOML format
cfg = dataio.read_config("./config-TEMPLATE.toml", schema=configs.pdp.PDPProjectConfig)
cfg

# COMMAND ----------

# Load human-friendly PDP feature names
features_table = dataio.read_features_table("assets/pdp/features_table.toml")

# COMMAND ----------

# MAGIC %md
# MAGIC # load artifacts

# COMMAND ----------

df = dataio.schemas.pdp.PDPLabeledDataSchema(
    dataio.read.from_delta_table(
        cfg.datasets.gold.silver.table_path,
        spark_session=spark,
    )
)
df.head()

# COMMAND ----------

model = dataio.models.load_mlflow_model(
    cfg.model.mlflow_model_uri,
    cfg.model.framework,
)
model

# COMMAND ----------

model_feature_names = model.named_steps["column_selector"].get_params()["cols"]
logging.info(
    "model uses %s features: %s", len(model_feature_names), model_feature_names
)

# COMMAND ----------

df_train = modeling.evaluation.extract_training_data_from_model(cfg.model.experiment_id)
if cfg.split_col:
    df_train = df_train.loc[df_train[cfg.split_col].eq("train"), :]
df_train.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # make predictions

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

pred_probs = modeling.inference.predict_probs(
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

with mlflow.start_run(run_id=cfg.model.run_id) as run:
    inference.shap_summary_plot(
        df_shap_values.loc[:, model_feature_names].to_numpy(),
        df_test.loc[:, model_feature_names],
        class_names=model.classes_,
        max_display=20,
        show=False,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # finalize results

# COMMAND ----------

with mlflow.start_run(run_id=cfg.model.run_id) as run:
    selected_features_df = inference.generate_ranked_feature_table(
        features,
        df_shap_values[model_feature_names].to_numpy(),
        features_table=features_table,
    )
selected_features_df

# COMMAND ----------

# Provide output using top features, SHAP values, and support scores
result = inference.select_top_features_for_display(
    features,
    unique_ids,
    pred_probs,
    df_shap_values[model_feature_names].to_numpy(),
    n_features=cfg.inference.num_top_features,
    features_table=features_table,
    needs_support_threshold_prob=cfg.inference.min_prob_pos_label,
)
result

# COMMAND ----------

# save sample advisor output dataset
dataio.write.to_delta_table(
    result, cfg.datasets.gold.advisor_output.table_path, spark_session=spark
)
