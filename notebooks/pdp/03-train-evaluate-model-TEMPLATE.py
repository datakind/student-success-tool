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

# install dependencies, most of which should come through our 1st-party SST package
# %pip install "student-success-tool==0.1.0"
# %pip install git+https://github.com/datakind/student-success-tool.git@develop
# %pip install git+https://github.com/datakind/student-success-tool.git@upgrade-package-deps

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------
