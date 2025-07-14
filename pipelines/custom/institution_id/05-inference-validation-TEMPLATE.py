# Databricks notebook source
# MAGIC %md
# MAGIC # SST Inference Validation
# MAGIC
# MAGIC The last step is our human-in-the-loop process before we send files back to institutions. It is a critical step to ensure that of our process to ensure high quality & consistency.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# WARNING: AutoML/mlflow expect particular packages with version constraints
# that directly conflicts with dependencies in our SST repo. As a temporary fix,
# we need to manually install a certain version of pandas and scikit-learn in order
# for our models to load and run properly.

# %pip install "student-success-tool==0.3.7"
# %restart_python

# COMMAND ----------

import os
import sys
import logging
from databricks.connect import DatabricksSession

import matplotlib.pyplot as plt

from student_success_tool import dataio, configs

if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    sys.path.insert(1, "../")

run_type = "predict"  # always predict

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# load config file
cfg = dataio.read_config(
    "./config-TEMPLATE.toml", schema=configs.custom.CustomProjectConfig
)
cfg

# COMMAND ----------

gold_prediction_table_name = getattr(
    cfg.datasets.gold["advisor_output"], f"{run_type}_table_path"
)
print(gold_prediction_table_name)

predict_df = dataio.read.from_delta_table(
    gold_prediction_table_name,
    spark_session=spark,
)
predict_df

# COMMAND ----------

# Extract student id from config
student_id_col = cfg.student_id_col

# Assert that final output has rows
assert predict_df.shape[0] > 0, "Final output has no rows."

# Report the number of students
num_students = len(predict_df[student_id_col])
print(f"Number of students: {num_students}")

# Assert that all STUDENT_ID values are unique
assert predict_df[student_id_col].is_unique, (
    "The student ID column contains duplicate values."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate distribution in comparison to training

# COMMAND ----------

predict_df["support_score"].hist(bins=50)
plt.title("Distribution of Support Scores on Inference Dataset")
plt.ylabel("# of Students")
plt.xlabel("support_score")

# COMMAND ----------

predict_df["support_score"].value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at Top Indicator Value Counts across all Students
# MAGIC If there's not many top features across the student population, it could be the sign of invalid predictions or an unhealthy model.

# COMMAND ----------

predict_df[[f"feature_{i}_name" for i in range(1, 6)]].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate that SHAP values are decreasing in magnitude for each student

# COMMAND ----------

assert all(
    all(
        abs(predict_df.loc[i, f"feature_{j}_importance"])
        >= abs(predict_df.loc[i, f"feature_{j + 1}_importance"])
        for j in range(1, 5)
    )
    for i in range(len(predict_df))
), (
    "Final output has invalid SHAP values across top ranked features for one or more students."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check whether there are feature values that are 'nan'

# COMMAND ----------

predict_df.loc[
    (predict_df[[f"feature_{i}_value" for i in range(1, 6)]] == "nan").any(axis=1)
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate examples for the most, median, and least risk students

# COMMAND ----------

# Maximum Risk Student
max_student = predict_df[
    predict_df["support_score"] == predict_df["support_score"].max()
]
max_student

# COMMAND ----------

# Minimum Risk Student
min_student = predict_df[
    predict_df["support_score"] == predict_df["support_score"].min()
]
min_student

# COMMAND ----------

# Median Risk Student
median_value = predict_df["support_score"].median()
median_student = predict_df.loc[
    (predict_df["support_score"] - median_value).abs().nsmallest(1).index
]
median_student
