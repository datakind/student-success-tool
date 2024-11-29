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

# install dependencies, most of which should come through our 1st-party SST package
# %pip install git+https://github.com/datakind/student-success-tool.git@develop

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import sys

import numpy as np
import pandas as pd
import seaborn as sb
from databricks.connect import DatabricksSession

from student_success_tool.analysis import pdp

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark_session = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## `student-success-intervention` hacks

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# HACK: insert our 1st-party (school-specific) code into PATH
if "../" not in sys.path:
    sys.path.insert(1, "../")

# TODO: specify school's subpackage
from analysis import *  # noqa: F403

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog config

# COMMAND ----------

catalog = "sst_dev"

# configure where data is to be read from / written to
inst_name = "SCHOOL"  # TODO: fill in school's name in Unity Catalog
schema = f"{inst_name}_silver"
catalog_schema = f"{catalog}.{schema}"
print(f"{catalog_schema=}")

# COMMAND ----------

# MAGIC %md
# MAGIC # read validated data

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** Whether you need to use the base or school-specific data schema was determined in the previous step, i.e. the data assessment / EDA notebook.

# COMMAND ----------

df_course = pdp.schemas.RawPDPCourseDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog_schema}.course_dataset_validated", spark_session=spark_session
    )
)
print(f"rows x cols = {df_course.shape}")
df_course.head()

# COMMAND ----------

df_cohort = pdp.schemas.RawPDPCohortDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog_schema}.cohort_dataset_validated", spark_session=spark_session
    )
)
print(f"rows x cols = {df_cohort.shape}")
df_cohort.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # transform and join datasets

# COMMAND ----------

# school-specific parameters that configure featurization code
# these should all be changed, as desired, and confirmed with client stakeholders
min_passing_grade = pdp.constants.DEFAULT_MIN_PASSING_GRADE
min_num_credits_full_time = pdp.constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME
course_level_pattern = pdp.constants.DEFAULT_COURSE_LEVEL_PATTERN
key_course_subject_areas = None
key_course_ids = None

# COMMAND ----------

df_student_terms = pdp.dataops.make_student_term_dataset(
    df_cohort,
    df_course,
    min_passing_grade=min_passing_grade,
    min_num_credits_full_time=min_num_credits_full_time,
    course_level_pattern=course_level_pattern,
    key_course_subject_areas=key_course_subject_areas,
    key_course_ids=key_course_ids,
)
df_student_terms

# COMMAND ----------

# take a peek at featurized columns -- it's a lot
df_student_terms.columns.tolist()

# COMMAND ----------

# save student-term dataset in unity catalog, as needed
# write_table_path = f"{catalog_schema}.student_term_dataset"
# pdp.dataio.write_data_to_delta_table(df_student_terms, write_table_path, spark_session)

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

# COMMAND ----------

# TODO: school-specific parameters that configure target variable code

# COMMAND ----------

# TODO: suitable function call for school's use case
# df_labeled = pdp.targets.*.make_labeled_dataset(...)
df_labeled = pd.DataFrame()

# COMMAND ----------

ax = sb.histplot(
    df_labeled.sort_values("cohort"),
    y="cohort",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

ax = sb.histplot(
    df_labeled.sort_values("cohort"),
    y="cohort",
    hue="target",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

# drop unwanted columns and mask values by time
df_labeled = pdp.dataops.clean_up_labeled_dataset_cols_and_vals(df_labeled)
df_labeled.shape

# COMMAND ----------

# save labeled dataset in unity catalog, as needed
# write_table_path = f"{catalog}.{schema}.labeled_dataset"
# pdp.dataio.write_data_to_delta_table(df_labeled, write_table_path, spark_session=spark_session)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature selection + splits

# COMMAND ----------

import mlflow

from student_success_tool.modeling.feature_selection import select_features

# databricks freaks out during feature selection if autologging isn't disabled
mlflow.autolog(disable=True)

# COMMAND ----------

non_feature_cols = [
    "student_guid",
    "target",
    "student_age",
    "race",
    "ethnicity",
    "gender",
    "first_gen",
]
df_labeled_selected = select_features(
    df_labeled,
    non_feature_cols=non_feature_cols,
    # TODO: configure these params as desired
    force_include_cols=[],
    incomplete_threshold=0.5,
    low_variance_threshold=0.0,
    collinear_threshold=10.0,
)
df_labeled_selected

# COMMAND ----------

# heads-up: AutoML doesn't preserve Student IDs in the training data!
# manually assigning splits lets us know which rows were in which partition
# when evaluating the model across student groups in the validation set
# default split strategy by AutoML is [0.6, 0.2, 0.2] for train/test/validation
np.random.seed(1)
df_labeled_selected = df_labeled_selected.assign(
    split=lambda df: np.random.choice(
        ["train", "test", "validate"],
        size=df.shape[0],
        p=[0.6, 0.2, 0.2],
    )
)
df_labeled_selected["split"].value_counts(normalize=True)

# COMMAND ----------

# save final modeling dataset in unity catalog, as needed
# write_table_path = f"{catalog}.{schema}.modeling_dataset"
# pdp.dataio.write_data_to_delta_table(df_labeled_selected, write_table_path, spark_session=spark_session)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature-target associations

# COMMAND ----------

target_assocs = pdp.eda.compute_pairwise_associations(
    df_labeled_selected, ref_col="target"
)
target_assocs

# COMMAND ----------

# MAGIC %md
# MAGIC # Wrap-up
# MAGIC
# MAGIC TODO

# COMMAND ----------
