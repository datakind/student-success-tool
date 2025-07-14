# Databricks notebook source
# MAGIC %md
# MAGIC # SST Prepare Modeling Dataset: [SCHOOL]
# MAGIC
# MAGIC Second step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: featurize the raw, validated data; configure and compute the target variable; perform feature selection; prepare train/test/validation splits; and inspect feature-target associations.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Data preparation for classification](https://docs.databricks.com/en/machine-learning/automl/classification-data-prep.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [SCHOOL WEBSITE](https://example.com)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, of which most/all should come through our 1st-party SST package

# %pip install "student-success-tool == 0.3.7"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import functools as ft
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd  # noqa: F401
import seaborn as sb
from databricks.connect import DatabricksSession

from student_success_tool import configs, dataio, eda, modeling, preprocessing
from student_success_tool.preprocessing import checkpoints, selection, targets

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
# MAGIC ## import school-specific code

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# insert our 1st-party (school-specific) code into PATH
if "../" not in sys.path:
    sys.path.insert(1, "../")

# TODO: specify school's subpackage
from pipelines import *  # noqa: F403

# COMMAND ----------

# project configuration should be stored in a config file in TOML format
cfg = dataio.read_config("./config-TEMPLATE.toml", schema=configs.pdp.PDPProjectConfig)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # load (+validate) raw data

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** The specifics of this load+validate process were determined previously in this project's "data assessment" notebook, including which schemas are needed to properly parse and validate the raw data. Also, the paths to the raw files should have been added to the project config. Update the next two cells accordingly.

# COMMAND ----------

raw_course_file_path = cfg.datasets.bronze.raw_course.file_path
df_course = dataio.pdp.read_raw_course_data(
    file_path=raw_course_file_path,
    schema=dataio.schemas.pdp.RawPDPCourseDataSchema,
    dttm_format="%Y%m%d.0",
)
df_course.head()

# COMMAND ----------

raw_cohort_file_path = cfg.datasets.bronze.raw_cohort.file_path
df_cohort = dataio.pdp.read_raw_cohort_data(
    file_path=raw_cohort_file_path, schema=dataio.schemas.pdp.RawPDPCohortDataSchema
)
df_cohort.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # featurize data

# COMMAND ----------

# load featurization params from the project config
feature_params = cfg.preprocessing.features.model_dump()

# COMMAND ----------

df_student_terms = preprocessing.pdp.make_student_term_dataset(
    df_cohort, df_course, **feature_params
)
df_student_terms

# COMMAND ----------

# take a peek at featurized columns -- it's a lot
df_student_terms.columns.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC # select students and compute targets

# COMMAND ----------

# MAGIC %md
# MAGIC A checkpoint, selection, and target function can be used together to make a labeled dataset for modeling. This involves selecting eligible students, subsetting to just the terms from which predictions are made, and computing target variables. _Which_ function depends primarily on the target to be computed: a failure to earn enough credits within a timeframe since initial enrollment, or a particular mid-way checkpoint (other targets pending). Input parameters will vary depending on the school and the target.
# MAGIC
# MAGIC Here's a concrete example below based on a retention model:
# MAGIC ```python
# MAGIC # school-specific parameters specified in config.toml
# MAGIC [preprocessing.selection]
# MAGIC student_criteria = { enrollment_type = "FIRST-TIME", credential_type_sought_year_1 = ["BACHELOR'S DEGREE", "ASSOCIATE'S Degree"], enrollment_intensity_first_term = ["FULL-TIME", "PART-TIME"] }
# MAGIC intensity_time_limits = { FULL-TIME = [3.0, "year"], PART-TIME = [6.0, "year"] }
# MAGIC
# MAGIC [preprocessing.checkpoint]
# MAGIC name = "end_of_first_term"
# MAGIC type_ = "first"
# MAGIC
# MAGIC [preprocessing.target]
# MAGIC name = "retention"
# MAGIC type_ = "retention"
# MAGIC max_academic_year = "2024"
# MAGIC
# MAGIC # creating checkpoint dataframe
# MAGIC df_ckpt = checkpoints.pdp.first_student_terms_within_cohort(
# MAGIC     df_student_terms,
# MAGIC     sort_cols=cfg.preprocessing.checkpoint.sort_cols,
# MAGIC     include_cols=cfg.preprocessing.checkpoint.include_cols,
# MAGIC )
# MAGIC # selecting students from student criteria
# MAGIC selected_students = selection.pdp.select_students_by_attributes(
# MAGIC     df_student_terms,
# MAGIC     student_id_cols=cfg.student_id_col,
# MAGIC     **cfg.preprocessing.selection.student_criteria,
# MAGIC )
# MAGIC # finding students who meet student criteria in checkpoints
# MAGIC df_labeled = pd.merge(df_ckpt, pd.Series(selected_students.index), how="inner", on=cfg.student_id_col)
# MAGIC
# MAGIC # computing target and adding it to df_labeled
# MAGIC target = targets.pdp.retention.compute_target(
# MAGIC     df_labeled,
# MAGIC     max_academic_year=cfg.preprocessing.target.max_academic_year,
# MAGIC )
# MAGIC df_labeled = pd.merge(df_labeled, target, how="inner", on=cfg.student_id_col)
# MAGIC ```
# MAGIC Check out the `checkpoint`, `selection`, & `targets` modules for options and more info.

# COMMAND ----------

# TODO: choose checkpoint function suitable for school's use case
# parameters should be specified in the config
df_ckpt = checkpoints.pdp.TODO(
    df_student_terms,
    sort_cols=cfg.preprocessing.checkpoint.sort_cols,
    include_cols=cfg.preprocessing.checkpoint.include_cols,
)
df_ckpt

# COMMAND ----------

# select students based on student criteria
selected_students = selection.pdp.select_students_by_attributes(
    df_student_terms,
    student_id_cols=cfg.student_id_col,
    **cfg.preprocessing.selection.student_criteria,
)

# merge checkpoint terms with selected students
df_labeled = pd.merge(
    df_ckpt, pd.Series(selected_students.index), how="inner", on=cfg.student_id_col
)
df_labeled

# COMMAND ----------

# TODO: choose target function suitable for school's use case
# parameters should be specified in the config
target = targets.pdp.TODO.compute_target(df_student_terms, **cfg.preprocessing.target)
df_labeled = pd.merge(df_labeled, target, how="inner", on=cfg.student_id_col)

print(df_labeled[cfg.target_col].value_counts())
print(df_labeled[cfg.target_col].value_counts(normalize=True))

# COMMAND ----------

ax = sb.histplot(
    df_labeled.sort_values("cohort"),
    y="cohort",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")
plt.show()
ax = sb.histplot(
    df_labeled.sort_values("cohort"),
    y="cohort",
    hue=cfg.target_col,
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")
plt.show()

# COMMAND ----------

# drop unwanted columns and mask values by time
df_preprocessed = preprocessing.pdp.clean_up_labeled_dataset_cols_and_vals(df_labeled)
df_preprocessed.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # splits and sample weights

# COMMAND ----------

try:
    splits = cfg.preprocessing.splits
    split_col = cfg.split_col
    sample_class_weight = cfg.preprocessing.sample_class_weight
    sample_weight_col = cfg.sample_weight_col
except AttributeError:
    splits = {"train": 0.6, "test": 0.2, "validate": 0.2}
    split_col = "split"
    sample_class_weight = "balanced"
    sample_weight_col = "sample_weight"

# COMMAND ----------

if splits:
    df_preprocessed = df_preprocessed.assign(
        **{
            split_col: ft.partial(
                modeling.utils.compute_dataset_splits, seed=cfg.random_state
            )
        }
    )
    print(df_preprocessed[split_col].value_counts(normalize=True))

# COMMAND ----------

if sample_class_weight:
    df_preprocessed = df_preprocessed.assign(
        **{
            sample_weight_col: ft.partial(
                modeling.utils.compute_sample_weights,
                target_col=cfg.target_col,
                class_weight=sample_class_weight,
            )
        }
    )
    print(df_preprocessed[sample_weight_col].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC # feature-target relationships

# COMMAND ----------

non_feature_cols = (
    [cfg.student_id_col]
    + (cfg.student_group_cols or [])
    + ([cfg.split_col] if cfg.split_col else [])
    + ([cfg.sample_weight_col] if cfg.sample_weight_col else [])
)

# COMMAND ----------

target_corrs = df_preprocessed.drop(
    columns=non_feature_cols + [cfg.target_col]
).corrwith(df_preprocessed[cfg.target_col], method="spearman", numeric_only=True)
print(target_corrs.sort_values(ascending=False).head(10))
print("...")
print(target_corrs.sort_values(ascending=False, na_position="first").tail(10))

# COMMAND ----------

target_assocs = eda.compute_pairwise_associations(
    df_preprocessed, ref_col=cfg.target_col, exclude_cols=non_feature_cols
)
print(target_assocs.sort_values(by="target", ascending=False).head(10))
print("...")
print(
    target_assocs.sort_values(by="target", ascending=False, na_position="first").tail(
        10
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC # wrap-up

# COMMAND ----------

# save preprocessed dataset
dataio.write.to_delta_table(
    df_preprocessed, cfg.datasets.silver.preprocessed.table_path, spark_session=spark
)

# COMMAND ----------

# MAGIC %md
# MAGIC - [ ] Submit a PR including this notebook and any changes in project config
