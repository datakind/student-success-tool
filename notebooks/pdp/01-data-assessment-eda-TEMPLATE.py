# Databricks notebook source
# MAGIC %md
# MAGIC # SST Data Assessment: [SCHOOL]
# MAGIC
# MAGIC First step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: validate the data, perform exploratory data analysis, and confirm alignment and/or clarify options on key technical decisions that will determine next steps.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - PDP raw data dictionaries: [Course dataset](https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary) and [Cohort dataset](https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [Databricks developer notebook utilities](https://docs.databricks.com/en/dev-tools/databricks-utils.html)
# MAGIC - [SCHOOL WEBSITE](https://example.com)

# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, most of which should come through our 1st-party SST package
# %pip install git+https://github.com/datakind/student-success-tool.git@develop

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import os
import sys

import missingno as msno
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
# MAGIC ### `student-success-intervention` hacks

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# HACK: insert our 1st-party code into PATH
if "../" not in sys.path:
    sys.path.insert(1, "../")

# COMMAND ----------

# MAGIC %md
# MAGIC # Read and Validate Raw Data

# COMMAND ----------

# configure where schools' raw data is stored in Unity Catalog
catalog = "sst_dev"
schema = "SCHOOL_bronze"
volume = "SCHOOL_bronze_file_volume"
path_volume = os.path.join("/Volumes", catalog, schema, volume)
path_table = f"{catalog}.{schema}"
print(f"{path_table=}")
print(f"{path_volume=}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### course dataset

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
fname_course = "SCHOOL_COURSE_AR_DEID_DTTM.csv"
fpath_course = os.path.join(path_volume, fname_course)
df_course_raw = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course, schema=None, dttm_format="%Y%m%d.0"
)
print(f"rows x cols = {df_course_raw.shape}")
df_course_raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Quick Checks:
# MAGIC - [ ] data exists where it should
# MAGIC - [ ] rows and columns are (broadly) as expected
# MAGIC - [ ] datetimes are consistently, legibly formatted

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
# TODO: update schema import path once https://github.com/datakind/student-success-tool/pull/19 is merged
base_course_schema = pdp.schemas.base.RawPDPCourseDataSchema

df_course = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course, schema=base_course_schema, dttm_format="%Y%m%d.0"
)
df_course

# COMMAND ----------

# MAGIC %md
# MAGIC If the above command works, and `df_course` is indeed a `pd.DataFrame` containing the validated + parsed PDP cohort dataset, then you're all set, and can skip ahead to the next section. If not, and this is instead a json blob of schema validation errors, then you'll need to iteratively develop a child data schema specific to SCHOOL. There are existing examples you can refer to in the `student-success-intervention` repo.

# COMMAND ----------


class RawSCHOOLCourseDataSchema(base_course_schema):
    # column-specific overrides, as needed, for example:
    # academic_year: pt.Series["string"] = pda.Field(nullable=True)
    # academic_term: pt.Series[pd.CategoricalDtype] = pda.Field(
    #     nullable=True,
    #     dtype_kwargs={
    #         "categories": ["FALL", "WINTER", "SPRING", "SUMMER"],
    #         "ordered": True,
    #     },
    # )
    ...


# COMMAND ----------

# MAGIC %md
# MAGIC If the raw data diverges from the PDP spec in more extreme/fundamental ways, you may need to develop a "preprocessing function" to bring it into line. A recurring example is in case of duplicate records. Refer to `pdp.dataio.read_raw_pdp_course_data_from_file()` for more guidance.

# COMMAND ----------


def course_data_preprocess_func(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    # make data PDP-compliant, for example:
    # df = df.drop_duplicates(subset=unique_columns)
    ...


# COMMAND ----------

# as needed
df_course = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course,
    dttm_format="%Y%m%d.0",
    schema=RawSCHOOLCourseDataSchema,
    preprocess_func=course_data_preprocess_func,
)
df_course

# COMMAND ----------

# MAGIC %md
# MAGIC `df_course` should be a properly validated and parsed data frame, ready for exploratory data analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### cohort dataset

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
fname_cohort = "SCHOOL_COHORT_AR_DEID_DTTM.csv"
fpath_cohort = os.path.join(path_volume, fname_cohort)
df_cohort_raw = pdp.dataio.read_raw_pdp_cohort_data_from_file(fpath_cohort, schema=None)
print(f"rows x cols = {df_cohort_raw.shape}")
df_cohort_raw.head()

# COMMAND ----------

# TODO: update schema import path once https://github.com/datakind/student-success-tool/pull/19 is merged
base_cohort_schema = pdp.schemas.base.RawPDPCohortDataSchema

df_cohort = pdp.dataio.read_raw_pdp_cohort_data_from_file(
    fpath_cohort, schema=base_cohort_schema
)
df_cohort

# COMMAND ----------

# MAGIC %md
# MAGIC Repeat the process from before, as needed: iteratively develop a school-specific cohort data schema and/or preprocessing function, then use them to produce the the validated/parsed data frame for analysis.

# COMMAND ----------

# # if needed
# df_cohort = pdp.dataio.read_raw_pdp_cohort_data_from_file(
#     fpath_cohort,
#     schema=RawSCHOOLCohortDataSchema,
#     preprocess_func=cohort_data_preprocess_func,
# )
# df_cohort

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - ...
# MAGIC
# MAGIC ### questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## summary stats

# COMMAND ----------

# decent, general-purpose summarization of a data frame
dbutils.data.summarize(df_course, precise=False)  # noqa: F821

# COMMAND ----------

# specific follow-ups, for example
# df_course["grade"].value_counts(normalize=True, dropna=False)
# df_course["delivery_method"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

dbutils.data.summarize(df_cohort, precise=True)  # noqa: F821

# COMMAND ----------

# specific follow-ups, for example
# df_course["cohort"].value_counts(normalize=True, dropna=False)
# df_course["enrollment_type"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - ...
# MAGIC
# MAGIC ### questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## null values

# COMMAND ----------

_ = msno.matrix(
    df_course.sort_values(by=["academic_year", "student_guid"], ignore_index=True),
    sparkline=False,
    labels=True,
)

# COMMAND ----------

df_course.isna().mean(axis="index").sort_values(ascending=False)

# COMMAND ----------

# confirm that *any* null course identifier means *all* null course identifiers
_ = msno.matrix(
    df_course.loc[df_course["course_prefix"].isna(), :].sort_values(
        by=["academic_year", "student_guid"], ignore_index=True
    ),
    sparkline=False,
    labels=True,
)

# COMMAND ----------

# check if null course identifiers are (almost) entirely for enrollments at other insts
df_course.loc[
    df_course["course_prefix"].isna(), "enrolled_at_other_institution_s"
].value_counts()

# COMMAND ----------

_ = msno.matrix(
    (
        df_course[df_course["enrolled_at_other_institution_s"].eq("N")].sort_values(
            by=["academic_year", "student_guid"]
        )
    )
)

# COMMAND ----------

_ = msno.matrix(
    df_cohort.sort_values(by=["cohort", "cohort_term"], ignore_index=True),
    sparkline=False,
    labels=True,
)

# COMMAND ----------

with pd.option_context("display.max_rows", 100):
    print(df_cohort.isna().mean(axis="index").sort_values(ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - ...
# MAGIC
# MAGIC ### questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## datasets join

# COMMAND ----------

df = pd.merge(
    df_cohort,
    df_course,
    on="student_guid",
    how="outer",
    suffixes=("_cohort", "_course"),
    indicator=True,
)
df["_merge"].value_counts()

# COMMAND ----------

# any patterns in mis-joined records?
df.loc[df["_merge"] != "both", :]

# COMMAND ----------

# which students don't appear in both datasets?
df.loc[df["_merge"] != "both", "student_guid"].unique().tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - ...
# MAGIC
# MAGIC ### questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## plots and key stats

# COMMAND ----------

(
    sb.histplot(
        df_cohort.sort_values("cohort"),
        y="cohort",
        hue="cohort_term",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

(
    sb.histplot(
        df_cohort,
        y="enrollment_type",
        hue="enrollment_intensity_first_term",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# same as plot above, only in cross-tab form
100 * pdp.eda.compute_crosstabs(
    df_cohort,
    "enrollment_type",
    "enrollment_intensity_first_term",
    normalize=True,
)

# COMMAND ----------

(
    sb.histplot(
        df_cohort,
        y="gender",
        hue="student_age",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# MAGIC %md
# MAGIC And so on, and so forth.

# COMMAND ----------

# if many records for other insts were found in the course dataset, you may want to filter them out from these plots
df_course_SCHOOL = df_course.loc[df_course["course_number"].notna(), :]

# COMMAND ----------

(
    sb.histplot(
        # df_course.sort_values(by="academic_year"),
        df_course_SCHOOL.sort_values(by="academic_year"),
        y="academic_year",
        hue="academic_term",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Course Enrollments")
)

# COMMAND ----------

num_distinct_courses = (
    df_course_SCHOOL["course_prefix"].str.cat(df_course["course_number"]).nunique()
)
print(f"number of distinct courses: {num_distinct_courses}")

# COMMAND ----------

ax = sb.histplot(
    pd.merge(
        df_course_SCHOOL.groupby("student_guid")
        .size()
        .rename("num_courses_enrolled")
        .reset_index(drop=False),
        df_cohort[["student_guid", "enrollment_intensity_first_term"]],
        on="student_guid",
        how="inner",
    ),
    x="num_courses_enrolled",
    hue="enrollment_intensity_first_term",
    multiple="stack",
    binwidth=5,
    edgecolor="white",
)
ax.set(xlabel="Number of courses enrolled (total)", ylabel="Number of students")
sb.move_legend(ax, loc="upper right", bbox_to_anchor=(1, 1))

# COMMAND ----------

jg = sb.jointplot(
    df_course_SCHOOL.groupby("student_guid").agg(
        {"number_of_credits_attempted": "sum", "number_of_credits_earned": "sum"}
    ),
    x="number_of_credits_attempted",
    y="number_of_credits_earned",
    kind="hex",
    joint_kws={"bins": "log"},
    marginal_kws={"edgecolor": "white"},
    ratio=4,
)
jg.refline(y=120.0)  # or whichever num credits earned is a relavent benchmark
jg.set_axis_labels("Number of Credits Attempted", "Number of Credits Earned")

# COMMAND ----------

# MAGIC %md
# MAGIC And so on, and so forth.

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - ...
# MAGIC
# MAGIC ### questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## correlations

# COMMAND ----------

df_course.corr(method="spearman", numeric_only=True)

# COMMAND ----------

df_cohort.corr(method="spearman", numeric_only=True)

# COMMAND ----------

# TODO: fill this out more thoroughly

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - ...
# MAGIC
# MAGIC ### questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC # Wrap-up

# COMMAND ----------

# MAGIC %md
# MAGIC - [ ] Add school-specific data schemas and/or preprocessing functions into the appropriate directory in the `student-success-intervention` repo
# MAGIC - TODO ...

# COMMAND ----------
