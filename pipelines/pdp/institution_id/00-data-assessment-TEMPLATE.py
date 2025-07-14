# Databricks notebook source
# MAGIC %md
# MAGIC # SST Data Assessment / EDA: [SCHOOL]
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
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, of which most/all should come through our 1st-party SST package

# %pip install "student-success-tool == 0.3.7"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import sys

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sb
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils

from student_success_tool import configs, dataio, eda

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

# TODO: specify school's subpackage here
from pipelines import *  # noqa: F403

# COMMAND ----------

# project configuration should be stored in a config file in TOML format
cfg = dataio.read_config("./config-TEMPLATE.toml", schema=configs.pdp.PDPProjectConfig)
cfg

# COMMAND ----------

# MAGIC %md
# MAGIC # read and validate raw data

# COMMAND ----------

# MAGIC %md
# MAGIC ## course dataset

# COMMAND ----------

# add actual path to school's raw course file into the project config
raw_course_file_path = cfg.datasets.bronze.raw_course.file_path

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
df_course_raw = dataio.pdp.read_raw_course_data(
    file_path=raw_course_file_path, schema=None, dttm_format="%Y%m%d.0"
)
df_course_raw.head()

# COMMAND ----------

df_course_raw.dtypes.value_counts()

# COMMAND ----------

df_course_raw["course_begin_date"].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Quick checks:
# MAGIC - [ ] data exists where it should
# MAGIC - [ ] rows and columns are (roughly) as expected
# MAGIC - [ ] datetimes are correctly parsed (they're consistently, legibly formatted)

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
df_course = dataio.pdp.read_raw_course_data(
    file_path=raw_course_file_path,
    schema=dataio.schemas.pdp.RawPDPCourseDataSchema,
    dttm_format="%Y%m%d.0",
)
df_course

# COMMAND ----------

# MAGIC %md
# MAGIC If the above command works, and `df_course` is indeed a `pd.DataFrame` containing the validated + parsed PDP cohort dataset, then you're all set, and can skip ahead to the next section. If not, and this is instead a json blob of schema errors, then you'll need to inspect those errors and iteratively develop school-specific overrides to handle them. There are existing examples you can refer to in the `student-success-intervention` repo if you're unsure.
# MAGIC
# MAGIC This will involve some ad-hoc exploratory work, depending on the schema errors. For example:
# MAGIC
# MAGIC ```python
# MAGIC # => any dupes? how to handle them??
# MAGIC >>> course_unique_cols = dataio.schemas.pdp.RawPDPCourseDataSchema.Config.unique
# MAGIC >>> df_course_raw.groupby(course_unique_cols).size().value_counts()
# MAGIC # categorical values with non-standard values?
# MAGIC >>> df_course_raw["co_requisite_course"].value_counts(dropna=False)
# MAGIC ```
# MAGIC
# MAGIC Depending on the results, you'll need a child data schema specific to SCHOOL and/or a preprocessing function to handle more significant deviations, such as duplicate rows. The former looks something like this:
# MAGIC
# MAGIC ```python
# MAGIC import pandas as pd
# MAGIC import pandera as pda
# MAGIC import pandera.typing as pt
# MAGIC
# MAGIC class RawSCHOOLCourseDataSchema(dataio.schemas.pdp.RawPDPCourseDataSchema):
# MAGIC     co_requisite_course: pt.Series[pd.CategoricalDtype] = pda.Field(
# MAGIC         nullable=True, dtype_kwargs={"categories": ["YES", "NO"]}
# MAGIC     )
# MAGIC ```
# MAGIC
# MAGIC The latter could be... many things. But fundamentally, it's a function that accepts a "raw", non-compliant dataframe and outputs a dataframe that is able to be validated and parsed by the schema. For example:
# MAGIC
# MAGIC ```python
# MAGIC def course_data_preprocess_func(df: pd.DataFrame, *, unique_cols: list[str]) -> pd.DataFrame:
# MAGIC     df_preproc = df.drop_duplicates(subset=unique_columns)
# MAGIC     if len(df_preproc) < len(df):
# MAGIC         logging.warning(
# MAGIC             "%s duplicate rows found and dropped",
# MAGIC             len(df) - len(df_preproc),
# MAGIC         )
# MAGIC     return df_preproc
# MAGIC ```
# MAGIC
# MAGIC So, in the worst-case scenario, your validation call will look something like this:
# MAGIC
# MAGIC ```python
# MAGIC import functools as ft
# MAGIC
# MAGIC df_course = dataio.pdp.read_raw_course_data(
# MAGIC     fpath_course,
# MAGIC     schema=RawSCHOOLCourseDataSchema,
# MAGIC     dttm_format="%Y%m%d.0",
# MAGIC     preprocess_func=ft.partial(
# MAGIC         course_data_preprocess_func,
# MAGIC         unique_cols=dataio.schemas.pdp.RawPDPCourseDataSchema.Config.unique
# MAGIC     ),
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC At this point, `df_course` should be a properly validated and parsed data frame, ready for exploratory data analysis.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## cohort dataset

# COMMAND ----------

# add actual path to school's raw cohort file into the project config
raw_cohort_file_path = cfg.datasets.bronze.raw_cohort.file_path

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
df_cohort_raw = dataio.pdp.read_raw_cohort_data(
    file_path=raw_cohort_file_path, schema=None
)
df_cohort_raw.head()

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
df_cohort = dataio.pdp.read_raw_cohort_data(
    file_path=raw_cohort_file_path, schema=dataio.schemas.pdp.RawPDPCohortDataSchema
)
df_cohort

# COMMAND ----------

# MAGIC %md
# MAGIC Repeat the process from before, as needed: iteratively develop a school-specific cohort data schema and/or preprocessing function, then use them to produce the the validated/parsed data frame for analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## takeaways / questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## STOP HERE!

# COMMAND ----------

# MAGIC %md
# MAGIC Before continuing on to EDA, now's a great time to do a couple things:
# MAGIC
# MAGIC - Copy any school-specific raw dataset schemas into a `schemas.py` file in the current working directory
# MAGIC - Copy any school-specific converter functions needed to coerce the raw data into a standardized form into a `dataio.py` file in the current working directory
# MAGIC - You can then reload the raw files using the necessary custom converter functions & schemas and then proceed with EDA below.

# COMMAND ----------

# MAGIC %md
# MAGIC # exploratory data analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## summary stats

# COMMAND ----------

# decent, general-purpose summarization of a data frame
dbutils.data.summarize(df_course, precise=False)

# COMMAND ----------

# specific follow-ups, for example
# df_course["academic_year"].value_counts(normalize=True, dropna=False)
# df_course["academic_term"].value_counts(normalize=True, dropna=False)
# df_course["grade"].value_counts(normalize=True, dropna=False)
# df_course["delivery_method"].value_counts(normalize=True, dropna=False)
# df_course["course_name"].value_counts(normalize=True, dropna=False).head(10)

# COMMAND ----------

dbutils.data.summarize(df_cohort, precise=True)

# COMMAND ----------

# specific follow-ups, for example
# df_cohort["cohort"].value_counts(normalize=True, dropna=False)
# df_cohort["enrollment_type"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways / questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## data validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### null values

# COMMAND ----------

_ = msno.matrix(
    df_course.sort_values(
        by=["academic_year", "enrolled_at_other_institution_s"], ignore_index=True
    ),
    sparkline=False,
    labels=True,
)

# COMMAND ----------

df_course.isna().mean(axis="index").sort_values(ascending=False)

# COMMAND ----------

# confirm that *any* null course identifier means *all* null course identifiers
# this is a known issue in PDP data -- course data is missing for students that transfer out
# we typically drop these records (see "filter invalid rows" section, below) since there is no info
# and they're not the population of interest
# definitely confirm with SCHOOL stakeholders during data assessment presentation
_ = msno.matrix(
    df_course.loc[df_course["course_prefix"].isna(), :].sort_values(
        by=["academic_year", cfg.student_id_col], ignore_index=True
    ),
    sparkline=False,
    labels=True,
)

# COMMAND ----------

# how many total?
num_missing_course_id = df_course["course_prefix"].isna().sum()
pct_missing_course_id = 100 * num_missing_course_id / len(df_course)
print(
    f"num rows missing course id: {num_missing_course_id} ({pct_missing_course_id:.1f}%)"
)

# COMMAND ----------

# check if null course identifiers are (almost) entirely for enrollments at other insts
df_course.loc[
    df_course["course_prefix"].isna(), "enrolled_at_other_institution_s"
].value_counts()

# COMMAND ----------

# how many for transfers vs non transfers?
missing_course = df_course.groupby(cfg.student_id_col).filter(
    lambda x: x["course_prefix"].isna().all()
)

missing_course.groupby(cfg.student_id_col)[
    "enrolled_at_other_institution_s"
].first().value_counts(dropna=False)

# COMMAND ----------

# here's the null visualization for rows with students not enrolled at other institutions
_ = msno.matrix(
    (
        df_course[df_course["enrolled_at_other_institution_s"].eq("N")].sort_values(
            by=["academic_year", cfg.student_id_col]
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
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ### datasets join

# COMMAND ----------

df = (
    pd.merge(
        df_cohort,
        df_course,
        on=cfg.student_id_col,
        how="outer",
        suffixes=("_cohort", "_course"),
        indicator=True,
    )
    # HACK: columns overlap on more than just student_guid
    # let's rename/drop a relevant few for convenience
    .rename(
        columns={
            "cohort_cohort": "cohort",
            "cohort_term_cohort": "cohort_term",
            "student_age_cohort": "student_age",
            "race_cohort": "race",
            "ethnicity_cohort": "ethnicity",
            "gender_cohort": "gender",
            "institution_id_cohort": "institution_id",
        }
    )
    .drop(
        columns=[
            "cohort_course",
            "cohort_term_course",
            "student_age_course",
            "race_course",
            "ethnicity_course",
            "gender_course",
            "institution_id_course",
        ]
    )
)
df["_merge"].value_counts()

# COMMAND ----------

# any patterns in mis-joined records?
df.loc[df["_merge"] != "both", :]

# COMMAND ----------

# which students don't appear in both datasets?
df.loc[df["_merge"] != "both", cfg.student_id_col].unique().tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ### logical consistency

# COMMAND ----------

# check for "pre-cohort" courses
(
    sb.histplot(
        (
            df.assign(
                has_pre_cohort_courses=lambda df: df["cohort"].gt(df["academic_year"])
            )
            .groupby(by=[cfg.student_id_col, "cohort"])
            .agg(has_pre_cohort_courses=("has_pre_cohort_courses", "any"))
            .reset_index(drop=False)
            .sort_values("cohort")
        ),
        y="cohort",
        hue="has_pre_cohort_courses",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

df_pre_cohort = df.loc[df["cohort"].gt(df["academic_year"]), :].assign(
    cohort_id=lambda df: df["cohort"].str.cat(df["cohort_term"], sep=" "),
    term_id=lambda df: df["academic_year"].str.cat(df["academic_term"], sep=" "),
)
df_pre_cohort[[cfg.student_id_col, "cohort_id", "term_id", "enrollment_type"]]

# COMMAND ----------

df_pre_cohort["enrollment_type"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### filter invalid rows(?)

# COMMAND ----------

# this is probably a filter you'll want to apply
# these courses known to be an issue w/ PDP data
df_course_filtered = df_course.loc[df_course["course_number"].notna(), :]
df_course_filtered.shape

# COMMAND ----------

# how many students remain?
df_course_filtered[cfg.student_id_col].nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC Filter cohort data respectively below:

# COMMAND ----------

df_cohort_filtered = df_cohort.loc[
    df_cohort[cfg.student_id_col].isin(df_course_filtered[cfg.student_id_col]), :
]
df_cohort_filtered

# COMMAND ----------

# MAGIC %md
# MAGIC ## plots and stats

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** You'll probably want to use the filtered dataframes for most of these plots, but not necessarily for all. Sometimes comparing the two can be instructive. For simplicity, all these example plots will just use the base data w/o extra data validation filtering applied. It's your call!

# COMMAND ----------

ax = sb.histplot(
    df_cohort.sort_values("cohort"),
    y="cohort",
    hue="cohort_term",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

num_cohorts = df_cohort["cohort"].nunique()
first_cohort, last_cohort = df_cohort["cohort"].min(), df_cohort["cohort"].max()
print(f"{num_cohorts} cohorts ({first_cohort} through {last_cohort})")

# COMMAND ----------

print(df_cohort["cohort_term"].value_counts(normalize=True, dropna=False), end="\n\n")
print(
    df_cohort["enrollment_type"].value_counts(normalize=True, dropna=False), end="\n\n"
)
print(
    df_cohort["enrollment_intensity_first_term"].value_counts(
        normalize=True, dropna=False
    ),
    end="\n\n",
)
print(
    df_cohort["credential_type_sought_year_1"].value_counts(
        normalize=True, dropna=False
    ),
    end="\n\n",
)

# COMMAND ----------

df_cohort["gpa_group_year_1"].describe()

# COMMAND ----------

# First year GPA by cohort and enrollment type
# to remove error bars, errorbar=None
ax = sb.barplot(
    df_cohort_filtered.sort_values(by="cohort").astype({"gpa_group_year_1": "Float32"}),
    x="cohort",
    y="gpa_group_year_1",
    estimator="mean",
    hue="enrollment_type",
    edgecolor="white",
)
ax.set(ylabel="Avg. GPA (Year 1)")
ax.legend(loc="lower left", title="Enrollment Type")

# COMMAND ----------

# First year GPA by cohort and credential type sought
# to remove error bars, errorbar=None
ax = sb.barplot(
    df_cohort_filtered.sort_values(by="cohort").astype({"gpa_group_year_1": "Float32"}),
    x="cohort",
    y="gpa_group_year_1",
    estimator="mean",
    hue="credential_type_sought_year_1",
    edgecolor="white",
)
ax.set(ylabel="Avg. GPA (Year 1)")
ax.legend(loc="lower left", title="Enrollment Intensity")

# COMMAND ----------

df_pct_creds_by_yr = pd.concat(
    [
        pd.DataFrame(
            {
                "year_of_enrollment": str(yr),
                "enrollment_type": df_cohort_filtered["enrollment_type"],
                "enrollment_intensity_first_term": df_cohort_filtered[
                    "enrollment_intensity_first_term"
                ],
                "pct_credits_earned": (
                    100
                    * df_cohort_filtered[f"number_of_credits_earned_year_{yr}"]
                    / df_cohort_filtered[f"number_of_credits_attempted_year_{yr}"]
                ),
            }
        )
        for yr in range(1, 5)
    ],
    axis="index",
    ignore_index=True,
)

# COMMAND ----------

df_pct_creds_by_yr["pct_credits_earned"] = df_pct_creds_by_yr[
    "pct_credits_earned"
].astype("float")

# median values
print(
    df_pct_creds_by_yr.groupby("year_of_enrollment")["pct_credits_earned"]
    .median()
    .value_counts(dropna=False)
)

# mean values
df_pct_creds_by_yr.groupby("year_of_enrollment")[
    "pct_credits_earned"
].mean().value_counts(dropna=False)

# COMMAND ----------

# Plot mean or median, based on above
(
    sb.barplot(
        df_pct_creds_by_yr,
        x="year_of_enrollment",
        y="pct_credits_earned",
        estimator="mean",
        edgecolor="white",
        errorbar=None,
    ).set(xlabel="Year of Enrollment", ylabel="Avg. % Credits Earned")
)

# Add percent labels on top of each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        (bar.get_x() + bar.get_width() / 2, height),
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.show()

# COMMAND ----------

ax = sb.histplot(
    df_course.sort_values(by="academic_year"),
    # df_course_filtered.sort_values(by="academic_year"),
    y="academic_year",
    hue="academic_term",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Course Enrollments")

# COMMAND ----------

num_ayears = df_course["academic_year"].nunique()
first_ayear, last_ayear = (
    df_course["academic_year"].min(),
    df_course["academic_year"].max(),
)
print(f"{num_ayears} academic years ({first_ayear} through {last_ayear})")

# COMMAND ----------

num_courses = (
    df_course["course_prefix"].str.cat(df_course["course_number"], sep=" ").nunique()
)
num_subjects = df_course["course_cip"].nunique()
print(f"{num_courses} distinct courses, {num_subjects} distinct subjects")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Student Demographics:

# COMMAND ----------

# enrollment type by enrollment intensity

ax = sb.histplot(
    df_cohort,
    y="enrollment_type",
    hue="enrollment_intensity_first_term",
    multiple="stack",
    discrete=True,
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

# same as plot above, only in cross-tab form
100 * eda.compute_crosstabs(
    df_cohort,
    "enrollment_type",
    "enrollment_intensity_first_term",
    normalize=True,
)

# COMMAND ----------

# degree seeking by enrollment type

(
    sb.histplot(
        df_cohort,
        y="enrollment_type",
        hue="credential_type_sought_year_1",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# student age by enrollment intensity

ax = sb.histplot(
    df_cohort,
    y="enrollment_intensity_first_term",
    hue="student_age",
    multiple="stack",
    discrete=True,
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

# student age by gender

ax = sb.histplot(
    df_cohort,
    y="gender",
    hue="student_age",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)

_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

# student gender by age
ax = sb.histplot(
    df_cohort[(df_cohort["gender"] == "F") | (df_cohort["gender"] == "M")],
    y="gender",
    hue="student_age",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)

_ = ax.set(xlabel="Number of Students")

# COMMAND ----------

# race by pell status
(
    sb.histplot(
        df_cohort,
        y="race",
        hue="pell_status_first_year",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Observe any systemic income disparities highlighted across races:

# COMMAND ----------

df_cohort[["race", "pell_status_first_year"]].groupby("race").value_counts(
    normalize=True, dropna=False
).sort_index() * 100

# COMMAND ----------

# first gen
(
    sb.histplot(
        df_cohort,
        y="first_gen",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# race by first_gen
(
    sb.histplot(
        df_cohort,
        y="race",
        hue="first_gen",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

# MAGIC %md
# MAGIC And so on, and so forth.

# COMMAND ----------

ax = sb.histplot(
    pd.merge(
        df_course.groupby(cfg.student_id_col)
        # df_course_filtered.groupby(cfg.student_id_col)
        .size()
        .rename("num_courses_enrolled")
        .reset_index(drop=False),
        df_cohort[[cfg.student_id_col, "enrollment_intensity_first_term"]],
        on=cfg.student_id_col,
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
    df_course.groupby(cfg.student_id_col).agg(
        {"number_of_credits_attempted": "sum", "number_of_credits_earned": "sum"}
    ),
    # df_course_filtered.groupby(cfg.student_id_col).agg(
    #     {"number_of_credits_attempted": "sum", "number_of_credits_earned": "sum"}
    # ),
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
# MAGIC ### takeaways / questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## variable associations

# COMMAND ----------

df_assoc_course = eda.compute_pairwise_associations(
    df_course,
    exclude_cols=[
        cfg.student_id_col,
        "institution_id",
        "student_age",
        "gender",
        "race",
        "ethnicity",
    ],
)
df_assoc_course

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 10))
sb.heatmap(
    df_assoc_course.astype(np.float32),
    xticklabels=df_assoc_course.columns,
    yticklabels=df_assoc_course.columns,
    vmin=0.0,
    vmax=1.0,
    ax=ax,
)
_ = ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
)

# COMMAND ----------

df_assoc_cohort = eda.compute_pairwise_associations(
    df_cohort,
    exclude_cols=[
        cfg.student_id_col,
        "institution_id",
        "student_age",
        "gender",
        "race",
        "ethnicity",
    ],
)
df_assoc_course

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 10))
sb.heatmap(
    df_assoc_cohort.astype(np.float32),
    xticklabels=df_assoc_cohort.columns,
    yticklabels=df_assoc_cohort.columns,
    vmin=0.0,
    vmax=1.0,
    ax=ax,
)
_ = ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways / questions
# MAGIC
# MAGIC - ...

# COMMAND ----------

# MAGIC %md
# MAGIC # wrap-up

# COMMAND ----------

# MAGIC %md
# MAGIC - [ ] If you haven't already, add school-specific data schemas and/or preprocessing functions into the appropriate directory in the [`student-success-intervention` repository](https://github.com/datakind/student-success-intervention)
# MAGIC - [ ] Add file paths for the raw course/cohort datasets to the project config file's `datasets[DATASET_NAME].raw_course` and `datasets[DATASET_NAME].raw_cohort` blocks
# MAGIC - [ ] Submit a PR including this notebook and any school-specific files added in order to run it
