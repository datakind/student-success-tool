# Databricks notebook source
# MAGIC %md
# MAGIC # SST Prepare Modeling Dataset: exuni3
# MAGIC
# MAGIC Second step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: featurize the raw, validated data; configure and compute the target variable; perform feature selection; prepare train/test/validation splits; and inspect feature-target associations.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Data preparation for classification](https://docs.databricks.com/en/machine-learning/automl/classification-data-prep.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [exuni3 WEBSITE](https://www.exuni3.edu/)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, most of which should come through our 1st-party SST package
#%pip install git+https://github.com/datakind/student-success-tool.git@develop

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys
import logging
import pandas as pd
import seaborn as sb
from databricks.connect import DatabricksSession

from student_success_tool.analysis import pdp
from student_success_tool.analysis.pdp.targets import failure_to_retain


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

# custom processing & cohort schema
from analysis.exuni3 import preprocessing, schemas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog config

# COMMAND ----------

catalog = "sst_dev"

# configure where data is to be read from / written to
inst_name = "exuni3"  # TODO: fill in school's name in Unity Catalog
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

df_cohort = schemas.Rawexuni3CohortDataSchema(
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
exuni3_course_level_pattern = '^(?P<course_level>\d)\d{2,}(?:[A-Z]{1,2})?(?:-\d+)?$'
key_course_subject_areas = None
key_course_ids = None

# COMMAND ----------

# filter invalid students
df_course, df_cohort = preprocessing.filter_invalid(df_course, df_cohort)

# COMMAND ----------

df_student_terms = pdp.dataops.make_student_term_dataset(
    df_cohort,
    df_course,
    min_passing_grade=min_passing_grade,
    min_num_credits_full_time=min_num_credits_full_time,
    course_level_pattern=exuni3_course_level_pattern,
    key_course_subject_areas=key_course_subject_areas,
    key_course_ids=key_course_ids,
)
df_student_terms

# COMMAND ----------

# Delta doesn't support arrays of NullType (e.g. [<NA>, <NA>]), so we need to overwrite the column
df_student_terms['course_subject_areas'] = pd.NA

# COMMAND ----------

# save student-term dataset in unity catalog, as needed
write_table_path = f"{catalog_schema}.student_term_dataset"
pdp.dataio.write_data_to_delta_table(df_student_terms, write_table_path, spark_session)

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

student_criteria = {
    "enrollment_type": "FIRST-TIME",
    "enrollment_intensity_first_term": ["FULL-TIME", "PART-TIME"],
    "credential_type_sought_year_1": ["Bachelor's Degree", "Associate's Degree"]

}
df_labeled = failure_to_retain.make_labeled_dataset(df_student_terms, student_criteria=student_criteria)

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

ax = sb.histplot(
    df_labeled.sort_values("cohort"),
    y="section_num_students_enrolled_mean",
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

# remove cumsum columns due to redundancy
# removing year_1 and prev_term as these are future columns (not relevant for our 1 term retention model)
future_cols = [col for col in df_labeled.columns if (('cum' in col) or ('year_1' in col) or ('prev_term' in col))]
df_labeled = df_labeled.drop(future_cols, axis=1)

# COMMAND ----------

df_labeled.isna().mean(axis="index").sort_values(ascending=False)[:15]

# COMMAND ----------

# MAGIC %md
# MAGIC # Wrap-up
# MAGIC
# MAGIC - With our second fall retention model, it appears that 30 or so selected features is enough to create a roughly 80% accurate model. 
# MAGIC - The data suggests that students are primarily not retaining into the second fall because:
# MAGIC   - They did not earn enough credits in their first term.
# MAGIC   - They attempted to take less than 10 credits of courses.
# MAGIC   - They failed at least one course or more. Though some students do not fail a course, and still may not retain.
# MAGIC   - They are part-time.
# MAGIC
# MAGIC - Many students who are missing data for "Program of Study - Term 1" or "Student Program of Study Area - Term 1" do not retain. We are assuming that this is because these students are "Undecided" in their major choice. We need to validate this with the institution prior to filling the NaN values.
