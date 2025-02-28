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
import os
import numpy as np
import pandas as pd
import seaborn as sb
from databricks.connect import DatabricksSession
import pandera as pda
import pandera.typing as pt
from IPython.display import display


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
inst_name = "inst_name"  # TODO: fill in school's name in Unity Catalog
bronze_schema = f"{inst_name}_bronze"
silver_schema = f"{inst_name}_silver"
gold_schema = f"{inst_name}_gold"

bronze_volume = os.path.join(
    "/Volumes", catalog, bronze_schema, f"{inst_name}_bronze_file_volume"
)
bronze_table = f"{catalog}.{bronze_schema}"
silver_table = f"{catalog}.{silver_schema}"

# COMMAND ----------

# MAGIC %md
# MAGIC # read validated data

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** Whether you need to use the base or school-specific data schema was determined in the previous step, i.e. the data assessment / EDA notebook.

# COMMAND ----------

df_course = pdp.schemas.RawPDPCourseDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{silver_table}.course_dataset_validated", spark_session=spark_session
    )
)
print(f"rows x cols = {df_course.shape}")
df_course.head()

# COMMAND ----------

display(df_course)

# COMMAND ----------

class CohortDataSchema(pdp.schemas.RawPDPCohortDataSchema):
    number_of_credits_attempted_year_1: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_earned_year_1: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_attempted_year_2: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_earned_year_2: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_attempted_year_3: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_earned_year_3: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_attempted_year_4: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    number_of_credits_earned_year_4: pt.Series[np.float32] = pda.Field(
        nullable=True, ge=0.0, raise_warning=True
    )
    first_gen: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y"]}
    )
    credential_type_sought_year_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={
            "categories": [
                "Associate's Degree",
                "Non Credential Program (Preparatory Coursework / Teach Certification)",
                "1-2 year certificate, less than Associates degree",
                "Undergraduate Certificate or Diploma Program",
                "Less than 1-year certificate, less than Associates degree",
                "UNKNOWN",
            ]
        },
    )
    completed_dev_math_y_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    completed_dev_english_y_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    enrollment_intensity_first_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )

    @pda.parser("enrollment_intensity_first_term")
    def convert_to_categorical(cls, series):
        return series.astype("category")


df_cohort = CohortDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{silver_table}.cohort_dataset_validated", spark_session=spark_session
    )
)
print(f"rows x cols = {df_cohort.shape}")
df_cohort.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-feature checks

# COMMAND ----------

assert df_course["course_number"].dropna().str.len().eq(3).all()
assert df_course["course_prefix"].dropna().str.len().eq(3).all()

# COMMAND ----------

# we also had students that took courses before their enrollment, but want to check if we have any of those still
df_check = df_course.copy()
df_check["academic_year_int"] = df_check["academic_year"].str[:4]
df_check["cohort_year_int"] = df_check["cohort"].str[:4]
display(df_check[df_check["cohort_year_int"] > df_check["academic_year_int"]])

# COMMAND ----------

# MAGIC %md
# MAGIC # transform and join datasets

# COMMAND ----------

min_passing_grade = pdp.constants.DEFAULT_MIN_PASSING_GRADE
min_num_credits_full_time = pdp.constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME
course_level_pattern = pdp.constants.DEFAULT_COURSE_LEVEL_PATTERN
key_course_subject_areas = None
key_course_ids = None
print(f"min_passing_grade: {min_passing_grade}")
print(f"min_num_credits_full_time: {min_num_credits_full_time}")
print(f"course_level_pattern: {course_level_pattern}")

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
# MAGIC ## School specific features
# MAGIC
# MAGIC - Math and English completion within the first 12 credits
# MAGIC - Is fully online learner
# MAGIC - Writing intensive in under 12 credits
# MAGIC
# MAGIC
# MAGIC Writing Intenstive Courses

# COMMAND ----------

# Create a feature that flags students that are fully online in a term
df_student_terms["term_online_student"] = df_student_terms[
    "frac_courses_delivery_method_o"
].eq(1.0)

# COMMAND ----------

writing_intensive_courses = [
    "COURSEPREFIX1",
    "COURSEPREFIX2",
    "COURSEPREFIX3",
    "COURSEPREFIX4",
    "COURSEPREFIX5",
    "COURSEPREFIX6",
]

# COMMAND ----------

# Create a feature that flags if a student has taken one of the courses deemed writing intensive within their first 12 credits
df_student_terms["writing_course_flag"] = df_student_terms["course_ids"].apply(
    lambda x: any(course_id in writing_intensive_courses for course_id in x)
)
df_student_terms["writing_course_wi_12_term"] = df_student_terms[
    "writing_course_flag"
] & (df_student_terms["num_credits_attempted_cumsum"] <= 12)
df_student_terms["writing_course_wi_12"] = (
    df_student_terms.groupby("student_guid")["writing_course_wi_12_term"]
    .transform("max")
    .astype(int)
)
df_student_terms.drop(
    ["writing_course_wi_12_term", "writing_course_flag"], axis=1, inplace=True
)

# COMMAND ----------

# Create a feature that flags if a student has taken a math or english course within their first 12 credits
# First, create a list of the English and Math course titles
df_course["course_pre_num"] = df_course["course_prefix"] + df_course["course_number"]
df_math_eng = df_course[
    (df_course["course_prefix"] == "ENG") | (df_course["course_prefix"] == "MTH")
]
course_pre_num_list = df_math_eng["course_pre_num"].tolist()
# Now flag if students took one of them
df_student_terms["math_eng_flag"] = df_student_terms["course_ids"].apply(
    lambda x: any(course_id in course_pre_num_list for course_id in x)
)
df_student_terms["math_eng_course_wi_12_term"] = df_student_terms["math_eng_flag"] & (
    df_student_terms["num_credits_attempted_cumsum"] <= 12
)
df_student_terms["math_eng_course_wi_12"] = (
    df_student_terms.groupby("student_guid")["math_eng_course_wi_12_term"]
    .transform("max")
    .astype(int)
)
df_student_terms.drop(
    ["math_eng_course_wi_12_term", "math_eng_flag"], axis=1, inplace=True
)

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

min_num_credits_checkin = 12.0
min_num_credits_target = 60.0
student_id_cols = "student_guid"
intensity_time_limits = [
    ("FULL-TIME", 2.0, "year"),
    ("PART-TIME", 5.0, "year"),
]
# I only have FIRST-TIME but may as well specify
student_criteria = {
    "enrollment_type": "FIRST-TIME",
    "credential_type_sought_year_1": "Associate's Degree",
}

# COMMAND ----------

df_labeled = pdp.targets.failure_to_earn_enough_credits_in_time_from_checkin.make_labeled_dataset(
    df_student_terms,
    min_num_credits_checkin=min_num_credits_checkin,
    min_num_credits_target=min_num_credits_target,
    student_criteria=student_criteria,
    student_id_cols=student_id_cols,
    intensity_time_lefts=intensity_time_limits,
)
df_labeled["target"] = df_labeled["target"].astype(int)

# COMMAND ----------

# Remember target=1 means they failed to "graduate"/earn enough credits
df_labeled["target"].value_counts()

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
    hue="enrollment_intensity_first_term",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)

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
df_labeled_features = pdp.dataops.clean_up_labeled_dataset_cols_and_vals(df_labeled)
df_labeled_features.shape

# COMMAND ----------

# We are checking in at 12 credits, so using much beyond the first year doesn't make sense anyhow. Thus, also the frac_credits_earned each year doesnt make sense.
drop_cols = (
    df_labeled.columns[df_labeled.columns.str.startswith("number_of_credits")].tolist()
    + df_labeled.columns[
        df_labeled.columns.str.startswith("frac_credits_earned")
    ].tolist()
    + df_labeled.columns[df_labeled.columns.str.contains("first_year_to_")].tolist()
    + df_labeled.columns[df_labeled.columns.str.contains("diff_term")].tolist()
    + df_labeled.columns[df_labeled.columns.str.contains("diff_prev_term")].tolist()
)

print(f"dropping additional {len(drop_cols)} columns")
df_labeled_features = df_labeled_features.drop(columns=drop_cols)
df_labeled_features.shape

# COMMAND ----------

print(
    df_labeled_features.columns[
        df_labeled_features.columns.str.contains("diff")
    ].tolist()
)

# COMMAND ----------

# Check number of terms and edit the features that are in the "future"
future_features = [
    "gpa_group_year_1",
    "program_of_study_year_1",
    "student_program_of_study_area_year_1",
    "student_program_of_study_changed_term_1_to_year_1",
    "student_program_of_study_area_changed_term_1_to_year_1",
    "diff_gpa_term_1_to_year_1",
    "num_courses_in_program_of_study_area_year_1",
    "num_courses_in_program_of_study_area_year_1_cumfrac",
]
df_labeled_features.loc[
    df_labeled_features["cumnum_terms_enrolled"] == 1, future_features
] = np.nan
df_labeled_features.loc[
    (df_labeled_features["cumnum_terms_enrolled"] == 2)
    & (df_labeled_features["cumnum_fall_spring_terms_enrolled"] == 1),
    future_features,
] = np.nan

# COMMAND ----------

# save labeled dataset in unity catalog, as needed
write_table_path = f"{catalog}.{silver_schema}.df_labeled_features"
pdp.dataio.write_data_to_delta_table(
    df_labeled_features, write_table_path, spark_session=spark_session
)

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
    "enrollment_type",
    "credential_type_sought_year_1",
    "term_is_pre_cohort",
]
df_labeled_selected = select_features(
    df_labeled_features,
    non_feature_cols=non_feature_cols,
    force_include_cols=[
        "gpa_group_term_1",
        "frac_courses_grade_is_failing_or_withdrawal",
        "course_grade_numeric_mean",
        "frac_courses_grade_above_section_avg",
        "math_eng_course_wi_12",
        "writing_course_wi_12",
        "term_online_student",
    ],
    incomplete_threshold=0.6,
    collinear_threshold=20.0,
)
display(df_labeled_selected)

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

import sklearn

df_labeled_selected = df_labeled_selected.assign(
    sample_weight=lambda df: sklearn.utils.compute_sample_weight(
        "balanced", df["target"]
    )
)
df_labeled_selected["sample_weight"].value_counts(normalize=True)

# COMMAND ----------

# save final modeling dataset in unity catalog, as needed
write_table_path = f"{catalog}.{silver_schema}.modeling_dataset"
pdp.dataio.write_data_to_delta_table(
    df_labeled_selected, write_table_path, spark_session=spark_session
)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature-target associations

# COMMAND ----------

df_labeled_selected.gender.value_counts()

# COMMAND ----------

target_assocs = pdp.eda.compute_pairwise_associations(
    df_labeled_selected, ref_col="target"
)
target_assocs
