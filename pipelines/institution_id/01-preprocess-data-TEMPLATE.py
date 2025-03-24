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

# install dependencies, most/all of which should come through our 1st-party SST package
# NOTE: it's okay to use 'develop' or a feature branch while developing this nb
# but when it's finished, it's best to pin to a specific version of the package
# %pip install "student-success-tool == 0.1.1"
# %pip install "git+https://github.com/datakind/student-success-tool.git@develop"

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
from databricks.sdk.runtime import dbutils
from py4j.protocol import Py4JJavaError

from student_success_tool import (
    dataio,
    eda,
    features,
    modeling,
    preprocessing,
    schemas,
    targets,
)

# COMMAND ----------

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# check if we're running this notebook as a "job" in a "workflow"
# if not, assume this is a training workflow using labeled data
try:
    run_type = dbutils.widgets.get("run_type")
    dataset_name = dbutils.widgets.get("dataset_name")
except Py4JJavaError:
    run_type = "train"
    dataset_name = "labeled"

logging.info("'%s' run of notebook using '%s' dataset", run_type, dataset_name)

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
from analysis import *  # noqa: F403

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
# MAGIC # load (+validate) raw data

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** The specifics of this load+validate process were determined previously in this project's "data assessment" notebook, including which schemas are needed to properly parse and validate the raw data. Also, the paths to the raw files should have been added to the project config. Update the next two cells accordingly.

# COMMAND ----------

raw_course_file_path = cfg.datasets[dataset_name].raw_course.file_path
df_course = dataio.pdp.read_raw_course_data(
    file_path=raw_course_file_path,
    schema=schemas.pdp.RawPDPCourseDataSchema,
    dttm_format="%Y%m%d.0",
)
print(f"rows x cols = {df_course.shape}")
df_course.head()

# COMMAND ----------

raw_cohort_file_path = cfg.datasets[dataset_name].raw_cohort.file_path
df_cohort = dataio.pdp.read_raw_cohort_data(
    file_path=raw_cohort_file_path, schema=schemas.pdp.RawPDPCohortDataSchema
)
print(f"rows x cols = {df_cohort.shape}")
df_cohort.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # featurize data

# COMMAND ----------

# TODO: load featurization params from the project config
# okay to hard-code it first then add it to the config later
try:
    feature_params = cfg.preprocessing.features.model_dump()
except AttributeError:
    feature_params = {
        "min_passing_grade": features.pdp.constants.DEFAULT_MIN_PASSING_GRADE,
        "min_num_credits_full_time": features.pdp.constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
        # NOTE: this pattern in particular may be something you need to change
        # schools have many different conventions for course numbering!
        "course_level_pattern": features.pdp.constants.DEFAULT_COURSE_LEVEL_PATTERN,
        "peak_covid_terms": features.pdp.constants.DEFAULT_PEAK_COVID_TERMS,
        "key_course_subject_areas": None,
        "key_course_ids": None,
    }

# COMMAND ----------

df_student_terms = preprocessing.pdp.dataops.make_student_term_dataset(
    df_cohort, df_course, **feature_params
)
df_student_terms

# COMMAND ----------

# take a peek at featurized columns -- it's a lot
df_student_terms.columns.tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC # filter students and compute target

# COMMAND ----------

# MAGIC %md
# MAGIC A single function call can be used to make a labeled dataset for modeling, which includes selecting eligible students, subsetting to just the terms from which predictions are made, and computing target variables. _Which_ function depends primarily on the target to be computed: a failure to earn enough credits within a timeframe since initial enrollment, or a particular mid-way checkpoint (other targets pending). Input parameters will vary depending on the school and the target.
# MAGIC
# MAGIC For example, here's how one could make a labeled dataset with the following setup: Filters for first-time students, enrolled either full- or part-time, seeking an Associate's degree; that earn at least 60 credits within 3 years of enrollment if full-time or 6 years of enrollment if part-time; making predictions from the first term for which they've earned 30 credits.
# MAGIC
# MAGIC ```python
# MAGIC # school-specific parameters that configure target variable code
# MAGIC min_num_credits_checkin = 30.0
# MAGIC min_num_credits_target = 60.0
# MAGIC student_criteria = {
# MAGIC     "enrollment_type": "FIRST-TIME",
# MAGIC     "enrollment_intensity_first_term": ["FULL-TIME", "PART-TIME"],
# MAGIC     "credential_type_sought_year_1": "Associate's Degree",
# MAGIC }
# MAGIC intensity_time_limits = [
# MAGIC     ("FULL-TIME", 3.0, "year"),
# MAGIC     ("PART-TIME", 6.0, "year"),
# MAGIC ]
# MAGIC
# MAGIC df_labeled = pdp.targets.failure_to_earn_enough_credits_in_time_from_enrollment.make_labeled_dataset(
# MAGIC     df_student_terms,
# MAGIC     min_num_credits_checkin=min_num_credits_checkin,
# MAGIC     min_num_credits_target=min_num_credits_target,
# MAGIC     student_criteria=student_criteria,
# MAGIC     intensity_time_limits=intensity_time_limits,
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC Check out the `pdp.targets` module for options and more info.
# MAGIC
# MAGIC **TODO(Burton?):**
# MAGIC - Separate filtering of student criteria + target variable calculation from the selection of featurized rows from which to make predictions
# MAGIC - Output labels as a series and join to features separately, rather than adding directly to the featurized data under the hood.

# COMMAND ----------

# TODO: load target params from the project config
# okay to hard-code it first then add it to the config later
try:
    target_params = cfg.preprocessing.target.params
    logging.info("target params: %s", target_params)
except AttributeError:
    target_params = {}

# COMMAND ----------

# TODO: run target-specific function suitable for school's use case
if run_type == "train":
    df_labeled = targets.pdp.TODO.make_labeled_dataset(
        df_student_terms, **target_params
    )
    print(df_labeled[cfg.target_col].value_counts())
    print(df_labeled[cfg.target_col].value_counts(normalize=True))
else:
    logging.info("run_type != 'train', so skipping target variable computation...")

# COMMAND ----------

if run_type == "train":
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
if run_type == "train":
    df_modeling = preprocessing.pdp.dataops.clean_up_labeled_dataset_cols_and_vals(
        df_labeled
    )
else:
    df_modeling = preprocessing.pdp.dataops.clean_up_labeled_dataset_cols_and_vals(
        df_student_terms
    )
df_modeling.shape

# COMMAND ----------

# MAGIC %md
# MAGIC # STOP HERE?

# COMMAND ----------

# MAGIC %md
# MAGIC Everything after this point is for labeled training data only. In case of a non-"train" run type, we'll just save the modeling dataset here and bail out of the notebook early.

# COMMAND ----------

# TODO: fill in the actual table path for school's preprocessed dataset
# okay to add it to project config now or later, whatever you prefer
# preprocessed_table_path = cfg.datasets[dataset_name].preprocessed.table_path
preprocessed_table_path = "CATALOG.INST_NAME_silver.modeling_dataset_DESCRIPTIVE_SUFFIX"

# COMMAND ----------

if run_type != "train":
    dataio.write.to_delta_table(
        df_modeling, preprocessed_table_path, spark_session=spark
    )
    dbutils.notebook.exit(
        f"'{dataset_name}' modeling dataset saved to {preprocessed_table_path}; "
        "exiting notebook..."
    )

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
    df_modeling = df_modeling.assign(
        **{
            split_col: ft.partial(
                modeling.utils.compute_dataset_splits, seed=cfg.random_state
            )
        }
    )
    print(df_modeling[split_col].value_counts(normalize=True))

# COMMAND ----------

if sample_class_weight:
    df_modeling = df_modeling.assign(
        **{
            sample_weight_col: ft.partial(
                modeling.utils.compute_sample_weights,
                target_col=cfg.target_col,
                class_weight=sample_class_weight,
            )
        }
    )
    print(df_modeling[sample_weight_col].value_counts(normalize=True))

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

target_corrs = df_modeling.drop(columns=non_feature_cols + [cfg.target_col]).corrwith(
    df_modeling[cfg.target_col], method="spearman", numeric_only=True
)
print(target_corrs.sort_values(ascending=False).head(10))
print("...")
print(target_corrs.sort_values(ascending=False, na_position="first").tail(10))

# COMMAND ----------

target_assocs = eda.compute_pairwise_associations(
    df_modeling, ref_col=cfg.target_col, exclude_cols=non_feature_cols
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
dataio.write.to_delta_table(df_modeling, preprocessed_table_path, spark_session=spark)

# COMMAND ----------

# MAGIC %md
# MAGIC - [ ] Update project config with paremters for the preprocessed dataset (`datasets[dataset_name].preprocessed`), feature and target definitions (`preprocessing.features`, `preprocessing.target.params`), as well as any splits / sample weight parameters (`preprocessing.splits`, `preprocessing.sample_class_weight`)
# MAGIC - [ ] Submit a PR including this notebook and any changes in project config

# COMMAND ----------
