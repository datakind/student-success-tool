# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC # SST Data Assessment / EDA: ExUni1
# MAGIC
# MAGIC First step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: validate the data, perform exploratory data analysis, and confirm alignment and/or clarify options on key technical decisions that will determine next steps.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - PDP raw data dictionaries: [Course dataset](https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary) and [Cohort dataset](https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [Databricks developer notebook utilities](https://docs.databricks.com/en/dev-tools/databricks-utils.html)
# MAGIC - [exuni1 website](https://www.exuni1.edu)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, most of which should come through our 1st-party SST package
# %pip install git+https://github.com/datakind/student-success-tool.git@develop
# %pip install "student-success-tool==0.1.0"

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import os
import sys

import matplotlib.pyplot as plt
import missingno as msno
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
except ImportError:
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

from pipelines import exuni1

# COMMAND ----------

# MAGIC %md
# MAGIC ## unity catalog config

# COMMAND ----------

# configure where data is to be read from / written to
inst_name = "exuni1"

catalog = "sst_dev"
read_schema = f"{inst_name}_bronze"
write_schema = f"{inst_name}_silver"

path_volume = os.path.join(
    "/Volumes", catalog, read_schema, f"{inst_name}_bronze_file_volume"
)
path_table = f"{catalog}.{read_schema}"
print(f"{path_table=}")
print(f"{path_volume=}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Read and Validate Raw Data

# COMMAND ----------

# MAGIC %md
# MAGIC ## course dataset

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
fpath_course = os.path.join(
    path_volume, "exuni1_ctc_COURSE_LEVEL_AR_DEID_20241029000414_oct30_2024.csv"
)
df_course_raw = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course, schema=None, dttm_format="%Y%m%d.0"
)
print(f"rows x cols = {df_course_raw.shape}")
df_course_raw.head()

# COMMAND ----------

df_course_raw.dtypes.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC Quick checks:
# MAGIC - [x] data exists where it should
# MAGIC - [x] rows and columns are (roughly) as expected
# MAGIC - [x] datetimes are correctly parsed (they're consistently, legibly formatted)

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
df_course = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course, schema=pdp.schemas.RawPDPCourseDataSchema, dttm_format="%Y%m%d.0"
)
df_course

# COMMAND ----------

course_unique_cols = pdp.schemas.RawPDPCourseDataSchema.Config.unique
df_course_raw.groupby(course_unique_cols).size().value_counts()
# => 274 dupe rows!

# COMMAND ----------

100 * 274 / len(df_course_raw)

# COMMAND ----------

dupe_rows = df_course_raw.loc[
    df_course_raw.duplicated(course_unique_cols, keep=False), :
].sort_values(
    by=course_unique_cols + ["number_of_credits_attempted"],
    ascending=False,
    ignore_index=True,
)
dupe_rows.head(10)
# => these looks like legit classes, oftentimes labs, that we want to keep separate

# COMMAND ----------

dupe_rows[
    [
        "student_guid",
        "academic_year",
        "academic_term",
        "course_prefix",
        "course_number",
        "section_id",
        "course_name",
    ]
].tail(6)

# COMMAND ----------


def course_data_preprocess_func(
    df: pd.DataFrame, *, unique_cols: list[str]
) -> pd.DataFrame:
    deduped_course_numbers = (
        df.loc[df.duplicated(unique_cols, keep=False), :]
        .sort_values(
            by=unique_cols + ["number_of_credits_attempted"],
            ascending=False,
            ignore_index=False,
        )
        .assign(
            grp_num=lambda df: df.groupby(unique_cols)["course_number"].transform(
                "cumcount"
            )
            + 1,
            course_number=lambda df: df["course_number"].str.cat(
                df["grp_num"].astype("string"), sep="-"
            ),
        )
        .loc[:, ["course_number"]]
    )
    logging.warning(
        "%s duplicate rows found; course numbers modified to avoid duplicates",
        len(deduped_course_numbers),
    )
    df.update(deduped_course_numbers)
    return df


# COMMAND ----------

import functools as ft

df_course = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course,
    schema=pdp.schemas.RawPDPCourseDataSchema,
    dttm_format="%Y%m%d.0",
    preprocess_func=ft.partial(
        course_data_preprocess_func,
        unique_cols=pdp.schemas.RawPDPCourseDataSchema.Config.unique,
    ),
)
df_course

# COMMAND ----------

# MAGIC %md
# MAGIC ## cohort dataset

# COMMAND ----------

fpath_cohort = os.path.join(
    path_volume, "exuni1_ctc_AR_DEIDENTIFIED_20241029000400_oct30_2024.csv"
)

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
df_cohort_raw = pdp.dataio.read_raw_pdp_cohort_data_from_file(fpath_cohort, schema=None)
print(f"rows x cols = {df_cohort_raw.shape}")
df_cohort_raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Quick checks:
# MAGIC - [ ] data exists where it should
# MAGIC - [ ] rows and columns are (roughly) as expected

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
df_cohort = pdp.dataio.read_raw_pdp_cohort_data_from_file(
    fpath_cohort, schema=pdp.schemas.RawPDPCohortDataSchema
)
df_cohort

# COMMAND ----------

df_cohort_raw["enrollment_intensity_first_term"].value_counts(
    normalize=True, dropna=False
)

# COMMAND ----------

df_cohort_raw["first_gen"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

df_cohort_raw["credential_type_sought_year_1"].value_counts(
    normalize=True, dropna=False
)

# COMMAND ----------

import pandas as pd
import pandera as pda
import pandera.typing as pt


class Rawexuni1CohortDataSchema(pdp.schemas.RawPDPCohortDataSchema):
    # NOTE: categories set in a parser, which forces "UNKNOWN" values to null
    enrollment_intensity_first_term: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )
    first_gen: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    # NOTE: categories set in a parser, which forces "UNKNOWN" values to null
    credential_type_sought_year_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True
    )

    @pda.parser("enrollment_intensity_first_term")
    def set_enrollment_intensity_first_term_categories(cls, series):
        return series.cat.set_categories(["FULL-TIME", "PART-TIME"])

    @pda.parser("credential_type_sought_year_1")
    def set_credential_type_sought_year_1_categories(cls, series):
        return series.cat.set_categories(["Bachelor's Degree"])


# COMMAND ----------

df_cohort = pdp.dataio.read_raw_pdp_cohort_data_from_file(
    fpath_cohort, schema=Rawexuni1CohortDataSchema
)
df_cohort

# COMMAND ----------

# MAGIC %md
# MAGIC ## takeaways / questions
# MAGIC
# MAGIC - Course data looks pretty good! Data loaded, looks sensible, and the standard dttm format worked as expected for the course begin/end date columns.
# MAGIC - These PDP datasets are "non-standard" in ways we've seen before for other schools. We're definitely going to want to update the base schemas to allow for these common deviations, e.g. nullable course identifiers (owing to other inst enrollment records), years to/of that go up to 8 instead of 7, etc. _Update:_ This has been done! The school-specific logic is much smaller now.
# MAGIC - Q: A small number of PDP course records have courses with identical identifying information but are, seemingly, separate courses. Many of them are science labs, taken at the same time as the main (non-lab) course. I decided to keep these courses separate, but disambiguated them by appending a 1 or 2 to the usual course number. Does that seem reasonable / unlikely to cause issues?
# MAGIC - Q: Do you start your academic year with the "FALL" or "SUMMER" term? (Looks like FALL, but wanted to double-check.)
# MAGIC - Q: In the cohort dataset, >99% of first-year credential types sought are "Bachelor's Degree", and <1% are "UNKNOWN". In what cases is "UNKNOWN" used? Do you offer any other non-Bachelor's credentials at your school?

# COMMAND ----------

# MAGIC %md
# MAGIC ## save validated data

# COMMAND ----------


pdp.dataio.write_data_to_delta_table(
    df_course,
    f"{catalog}.{write_schema}.course_dataset_validated",
    spark_session=spark_session,
)

# COMMAND ----------

pdp.dataio.write_data_to_delta_table(
    df_cohort,
    f"{catalog}.{write_schema}.cohort_dataset_validated",
    spark_session=spark_session,
)

# COMMAND ----------

# MAGIC %md
# MAGIC # exploratory data analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## read validated data
# MAGIC
# MAGIC (so you don't have to execute the validation process more than once)

# COMMAND ----------

df_course = pdp.schemas.RawPDPCourseDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog}.{write_schema}.course_dataset_validated",
        spark_session=spark_session,
    )
)
df_course.shape

# COMMAND ----------

df_cohort = exuni1.schemas.Rawexuni1CohortDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog}.{write_schema}.cohort_dataset_validated",
        spark_session=spark_session,
    )
)
df_cohort.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## summary stats

# COMMAND ----------

# decent, general-purpose summarization of a data frame
dbutils.data.summarize(df_course, precise=False)  # noqa: F821

# COMMAND ----------

df_course["grade"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

dbutils.data.summarize(df_cohort, precise=True)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways / questions
# MAGIC
# MAGIC - Course grade comes in numeric values only, in 0.5 increments. This doesn't align with how letter grades are usually converted into numeric values (e.g. A- => 3.7, B+ => 3.3), and it doesn't include any P(ass)/F(ail)/W(ithdrawal)/I(ncomplete) categorical values. Is this expected? Can you explain how these numbers are generated?
# MAGIC - 75% of course delivery method values are null, and only from earlier academic years. Is this expected?
# MAGIC - The number of credits attempted/earned per year has increasing numbers of 0 values going from year 1 to year 4. Is that from students dropping out over time?
# MAGIC - The `years_to_bachelors_at_cohort_inst` column is entirely null; typically, this gives the integer number of years that a student takes to earn their Bachelor's degree, while null values indicate that a student has not (yet) earned this credential. Assuming at least some of your students graduate, this data doesn't make sense. Can you explain what's going on?
# MAGIC     - In contrast, `years_to_bachelor_at_other_inst` is not entirely null, and its values are in line with expectations.
# MAGIC - Similarly, `time_to_credential` is also entirely null. There are no graduations in your data.
# MAGIC - Math/English/Reading placement values are entirely "C", indicating that students are "college-ready". Is this expected? Is it a requirement for enrollment, or just not tracked?
# MAGIC - There are more than twice as many students identifying as FEMALE compared to MALE, a surprisingly large gender imbalance. Is this expected?

# COMMAND ----------

# MAGIC %md
# MAGIC ## data validation

# COMMAND ----------

df_course["delivery_method"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

df_course.loc[df_course["delivery_method"].notna(), "academic_year"].min()

# COMMAND ----------

print(df_cohort["math_placement"].value_counts(dropna=False))
print(df_cohort["english_placement"].value_counts(dropna=False))

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

# => confirm that *any* null course identifier means *all* null course identifiers
# this is a known issue in PDP data -- course data is missing for students that transfer out
# we typically drop these records (see "filter invalid rows" section, below) since there is no info
# and they're not the population of interest
# definitely confirm with SCHOOL stakeholders during data assessment presentation
_ = msno.matrix(
    df_course.loc[df_course["course_prefix"].isna(), :].sort_values(
        by=["academic_year", "student_guid"], ignore_index=True
    ),
    sparkline=False,
    labels=True,
)

# COMMAND ----------

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

(
    df_course.loc[df_course["enrolled_at_other_institution_s"].eq("Y"), :]
    .groupby(by=["academic_year"])
    .agg(nunique_students=("student_guid", "nunique"))
)

# COMMAND ----------

# here's the null visualization for rows with students not enrolled at other institutions
_ = msno.matrix(
    (
        df_course[df_course["enrolled_at_other_institution_s"].eq("N")].sort_values(
            by=["academic_year", "academic_term"]
        )
    ),
    sparkline=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - Course dataset is as we've seen before: Rows representing enrollments at *other* institutions have mostly null values. But the rest are mostly/entirely populated.

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

df_cohort.loc[:, ["program_of_study_term_1", "program_of_study_year_1"]].value_counts(
    normalize=True, dropna=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - Why does `program_of_study_term_1` switch from all null, to fully populated, back to null, then back to populated over time?
# MAGIC - Missing indicators for graduation is expected; according to SME, ExUni1 graduation data is stored with another campus' data, so it's not available here.

# COMMAND ----------

# MAGIC %md
# MAGIC ### datasets join

# COMMAND ----------

df = (
    pd.merge(
        df_cohort,
        df_course,
        on="student_guid",
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
unmatched_student_ids = df.loc[df["_merge"] != "both", "student_guid"].unique().tolist()
unmatched_student_ids

# COMMAND ----------

# MAGIC %md
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - Good! Almost perfect join between cohort and course datasets: Just 2 students found only in cohort dataset...
# MAGIC   - Q: Can we safely drop these mismatched students?

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
            .groupby(by=["student_guid", "cohort"])
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
df_pre_cohort[["student_guid", "cohort_id", "term_id", "enrollment_type"]].head(10)

# COMMAND ----------

df_pre_cohort[["cohort_id", "term_id"]].value_counts().head(25)

# COMMAND ----------

df_pre_cohort["enrollment_type"].value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### filter invalid rows(?)

# COMMAND ----------

df_cohort_valid = df_cohort.loc[
    ~df_cohort["student_guid"].isin(unmatched_student_ids), :
]
df_cohort_valid.shape

# COMMAND ----------

df_course_valid = df_course.loc[df_course["course_number"].notna(), :]
df_course_valid.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## plots and stats

# COMMAND ----------

# MAGIC %md
# MAGIC ### "key stats"

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

ax = sb.histplot(
    df_course.sort_values(by="academic_year"),
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

(
    pd.merge(
        df_cohort[["student_guid", "enrollment_intensity_first_term"]],
        df_course[["student_guid", "academic_year"]],
        on="student_guid",
        how="inner",
    )
    .loc[lambda df: df["enrollment_intensity_first_term"].eq("PART-TIME"), :]
    .groupby(by=["student_guid", "academic_year"])
    .size()
    .mean()
)

# COMMAND ----------

df_course["grade"].astype("Float32").mean()

# COMMAND ----------

# MAGIC %md
# MAGIC ### "key stats redux"

# COMMAND ----------

(
    sb.histplot(
        df_cohort_valid.sort_values("cohort"),
        y="cohort",
        hue="cohort_term",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

df_cohort_valid.sort_values("cohort").groupby("cohort").size()

# COMMAND ----------

num_cohorts = df_cohort_valid["cohort"].nunique()
first_cohort, last_cohort = (
    df_cohort_valid["cohort"].min(),
    df_cohort_valid["cohort"].max(),
)
print(f"{num_cohorts} cohorts ({first_cohort} through {last_cohort})")

# COMMAND ----------

print(
    df_cohort_valid["cohort_term"].value_counts(normalize=True, dropna=False),
    end="\n\n",
)
print(
    df_cohort_valid["enrollment_type"].value_counts(normalize=True, dropna=False),
    end="\n\n",
)
print(
    df_cohort_valid["enrollment_intensity_first_term"].value_counts(
        normalize=True, dropna=False
    ),
    end="\n\n",
)
print(
    df_cohort_valid["credential_type_sought_year_1"].value_counts(
        normalize=True, dropna=False
    ),
    end="\n\n",
)
print(df_cohort_valid["gpa_group_year_1"].describe())

# COMMAND ----------

ax = sb.histplot(
    df_course_valid.sort_values(by="academic_year"),
    y="academic_year",
    hue="academic_term",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
)
_ = ax.set(xlabel="Number of Course Enrollments")

# COMMAND ----------

print(f"{len(df_course_valid)} total course enrollments")

num_ayears = df_course_valid["academic_year"].nunique()
first_ayear, last_ayear = (
    df_course_valid["academic_year"].min(),
    df_course_valid["academic_year"].max(),
)
print(f"{num_ayears} academic years ({first_ayear} through {last_ayear})")

num_courses = (
    df_course_valid["course_prefix"]
    .str.cat(df_course_valid["course_number"], sep=" ")
    .nunique()
)
num_subjects = df_course_valid["course_cip"].nunique()
print(f"{num_courses} distinct courses, {num_subjects} distinct subjects")

avg_course_grade = df_course_valid["grade"].astype("Float32").mean()
print(f"avg course grade (all time) of {avg_course_grade:.1f}")

(
    pd.merge(
        df_cohort_valid[["student_guid", "enrollment_intensity_first_term"]],
        df_course_valid[["student_guid", "academic_year"]],
        on="student_guid",
        how="inner",
    )
    .loc[lambda df: df["enrollment_intensity_first_term"].eq("PART-TIME"), :]
    .groupby(by=["student_guid", "academic_year"])
    .size()
    .mean()
)

# COMMAND ----------

df_course_valid["academic_term"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

df_course_valid.loc[
    df_course_valid["academic_term"].eq("SUMMER"), "academic_year"
].max()

# COMMAND ----------

# MAGIC %md
# MAGIC ### more plots

# COMMAND ----------

_ = sb.histplot(
    df_cohort_valid,
    y="gender",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
).set(xlabel="Number of Students")

# COMMAND ----------

df_cohort_valid["gender"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

_ = sb.histplot(
    df_cohort_valid,
    y="student_age",
    hue="enrollment_type",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
).set(xlabel="Number of Students")

# COMMAND ----------

df_cohort_valid["student_age"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

_ = sb.histplot(
    df_cohort_valid,
    y="race",
    hue="ethnicity",
    hue_order=["N", "H"],
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
).set(xlabel="Number of Students")

# COMMAND ----------

df_cohort_valid["race"].value_counts(normalize=False, dropna=False)

# COMMAND ----------

df_cohort_valid["ethnicity"].value_counts(normalize=False, dropna=False)

# COMMAND ----------

_ = sb.histplot(
    df_cohort_valid,
    y="ethnicity",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
).set(xlabel="Number of Students")

# COMMAND ----------

_ = sb.histplot(
    df_cohort_valid,
    y="first_gen",
    multiple="stack",
    shrink=0.75,
    edgecolor="white",
).set(xlabel="Number of Students")

# COMMAND ----------

ax = sb.histplot(
    # NOTE: this should re-order categories but doesn't, bc pandas is a mess
    df_cohort_valid.sort_values(
        by="enrollment_type",
        key=lambda s: s.replace({"FIRST-TIME": 1, "TRANSFER-IN": 2, "RE-ADMIT": 3}),
    ),
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
100 * pdp.eda.compute_crosstabs(
    df_cohort_valid,
    "enrollment_type",
    "enrollment_intensity_first_term",
    normalize=True,
)

# COMMAND ----------

100 * df_cohort_valid["enrollment_type"].value_counts(normalize=True, dropna=False)

# COMMAND ----------

100 * df_cohort_valid["enrollment_intensity_first_term"].value_counts(
    normalize=True, dropna=False
)

# COMMAND ----------

(
    sb.barplot(
        df_cohort_valid.sort_values(by="cohort").astype(
            {"gpa_group_year_1": "Float32"}
        ),
        x="cohort",
        y="gpa_group_year_1",
        estimator="mean",
        hue="enrollment_type",
        edgecolor="white",
        err_kws={"alpha": 0.9, "linewidth": 1.5},
    ).set(ylabel="Avg. GPA (Year 1)")
)

# COMMAND ----------

(
    sb.barplot(
        df_cohort_valid.sort_values(by="cohort").astype(
            {"gpa_group_year_1": "Float32"}
        ),
        x="cohort",
        y="gpa_group_year_1",
        estimator="mean",
        hue="enrollment_intensity_first_term",
        edgecolor="white",
        err_kws={"alpha": 0.9, "linewidth": 1.5},
    ).set(ylabel="Avg. GPA (Year 1)")
)

# COMMAND ----------

df_pct_creds_by_yr = pd.concat(
    [
        pd.DataFrame(
            {
                "year_of_enrollment": str(yr),
                "enrollment_type": df_cohort_valid["enrollment_type"],
                "enrollment_intensity_first_term": df_cohort_valid[
                    "enrollment_intensity_first_term"
                ],
                "pct_credits_earned": (
                    100
                    * df_cohort_valid[f"number_of_credits_earned_year_{yr}"]
                    / df_cohort_valid[f"number_of_credits_attempted_year_{yr}"]
                ),
            }
        )
        for yr in range(1, 5)
    ],
    axis="index",
    ignore_index=True,
)
(
    sb.barplot(
        df_pct_creds_by_yr,
        x="year_of_enrollment",
        y="pct_credits_earned",
        # hue="enrollment_intensity_first_term",
        estimator="mean",
        edgecolor="white",
        err_kws={"alpha": 0.9, "linewidth": 1.5},
    ).set(xlabel="Year of Enrollment", ylabel="Avg. % Credits Earned")
)

# COMMAND ----------

df_pct_credits_earned = df_cohort_valid.assign(
    num_credits_attempted=lambda df: (
        df[[f"number_of_credits_attempted_year_{yr}" for yr in range(1, 5)]].sum(
            axis="columns"
        )
    ),
    num_credits_earned=lambda df: (
        df[[f"number_of_credits_earned_year_{yr}" for yr in range(1, 5)]].sum(
            axis="columns"
        )
    ),
    pct_credits_earned=lambda df: 100
    * df["num_credits_earned"]
    / df["num_credits_attempted"],
).loc[:, ["enrollment_type", "enrollment_intensity_first_term", "pct_credits_earned"]]

avg_pct_credits_earned = (
    df_pct_credits_earned["pct_credits_earned"].astype("float").mean()
)
pct_students_earn_all_credits = (
    100 * df_pct_credits_earned["pct_credits_earned"].eq(100).mean()
)
print(f"Avgerage % credits earned: {avg_pct_credits_earned:.1f}%")
print(f"% of students who earn every credit: {pct_students_earn_all_credits:.1f}%")

# COMMAND ----------

(
    sb.histplot(
        df_cohort.astype({"retention": "string"}),
        y="retention",
        hue="enrollment_type",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

df_cohort.groupby("enrollment_type")["retention"].mean()

# COMMAND ----------

(
    sb.histplot(
        df_cohort.astype({"retention": "string"}),
        y="retention",
        hue="enrollment_intensity_first_term",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

df_course_valid["course_prefix"].str.cat(
    df_course["course_number"], sep=" "
).value_counts().head(25)

# COMMAND ----------

ax = sb.histplot(
    pd.merge(
        df_course_valid.groupby("student_guid")
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
    binrange=(0, 80),
    edgecolor="white",
)
ax.set(xlabel="Number of courses enrolled (total)", ylabel="Number of students")
sb.move_legend(ax, loc="upper right", bbox_to_anchor=(1, 1))

# COMMAND ----------

ax = sb.histplot(
    pd.merge(
        (
            df_course_valid.assign(
                term_id=lambda df: df["academic_year"].str.cat(
                    df["academic_term"], sep=" "
                )
            )
            .groupby("student_guid", as_index=False)
            .agg(num_terms_enrolled=("term_id", "nunique"))
        ),
        df_cohort[["student_guid", "enrollment_intensity_first_term"]],
        on="student_guid",
        how="inner",
    ),
    x="num_terms_enrolled",
    hue="enrollment_intensity_first_term",
    multiple="stack",
    binwidth=1,
    edgecolor="white",
)
ax.set(xlabel="Number of terms enrolled (total)", ylabel="Number of students")
sb.move_legend(ax, loc="upper right", bbox_to_anchor=(1, 1))

# COMMAND ----------

ax = sb.histplot(
    pd.merge(
        (
            df_course_valid.groupby("student_guid", as_index=False).agg(
                num_years_enrolled=("academic_year", "nunique")
            )
        ),
        df_cohort[["student_guid", "enrollment_intensity_first_term"]],
        on="student_guid",
        how="inner",
    ),
    x="num_years_enrolled",
    hue="enrollment_intensity_first_term",
    multiple="stack",
    binwidth=1,
    edgecolor="white",
)
ax.set(xlabel="Number of years enrolled (total)", ylabel="Number of students")
sb.move_legend(ax, loc="upper right", bbox_to_anchor=(1, 1))

# COMMAND ----------

jg = sb.jointplot(
    df_course_valid.groupby("student_guid").agg(
        {"number_of_credits_attempted": "sum", "number_of_credits_earned": "sum"}
    ),
    x="number_of_credits_attempted",
    y="number_of_credits_earned",
    kind="hex",
    joint_kws={"bins": "log"},
    marginal_kws={"edgecolor": "white"},
    ratio=4,
    xlim=(0, 175),
    ylim=(0, 175),
)
jg.refline(y=120.0)
jg.set_axis_labels("Number of Credits Attempted", "Number of Credits Earned")

# COMMAND ----------

result = (
    (
        100
        * df_cohort_valid["number_of_credits_earned_year_1"]
        / (1e-3 + df_cohort_valid["number_of_credits_attempted_year_1"])
    )
    .replace([np.inf, -np.inf], np.nan)
    .mean()
)
print(f"avg. percentage of credits earned first year = {result:.2f}%")

# COMMAND ----------

(
    df_course_valid.groupby("student_guid")
    .agg({"number_of_credits_attempted": "sum", "number_of_credits_earned": "sum"})
    .assign(
        num_credits_lost=lambda df: df["number_of_credits_attempted"]
        - df["number_of_credits_earned"]
    )
    .loc[:, "num_credits_lost"]
    .eq(0.0)
).value_counts(normalize=True)
# => 36% of students earn every credit they attempt

# COMMAND ----------

ax = sb.histplot(
    pd.merge(
        (
            df_course_valid.groupby(
                ["student_guid", "academic_year", "academic_term"],
                observed=True,
                as_index=False,
            )
            .size()
            .rename(columns={"size": "num_courses_per_term"})
            .groupby("student_guid", as_index=False)
            .agg(avg_num_courses_per_term=("num_courses_per_term", "mean"))
        ),
        df_cohort[["student_guid", "enrollment_intensity_first_term"]],
        on="student_guid",
        how="inner",
    ),
    x="avg_num_courses_per_term",
    hue="enrollment_intensity_first_term",
    multiple="stack",
    binwidth=0.5,
    edgecolor="white",
)
ax.set(xlabel="Avg. number of courses enrolled per term", ylabel="Number of students")
sb.move_legend(ax, loc="upper left", bbox_to_anchor=(0, 1))

# COMMAND ----------

ax = sb.histplot(
    df_course_valid.astype({"grade": "Float32"}),
    x="grade",
    hue="delivery_method",
    hue_order=["F", "O", "H"],
    multiple="stack",
    binwidth=0.5,
    discrete=True,
    edgecolor="white",
)
ax.set(xlabel="Course grade", ylabel="Number of course enrollments")
ax.legend(title="Delivery method", labels=["Hybrid", "Online", "Face-to-face"])

# COMMAND ----------

print("grade percentage totals per delivery method")
100 * pdp.eda.compute_crosstabs(
    df_course_valid.astype({"grade": "Float32"}),
    "grade",
    "delivery_method",
    normalize="columns",
)

# COMMAND ----------

100 * pdp.eda.compute_crosstabs(
    df_course_valid.astype({"grade": "Float32"}),
    "grade",
    "delivery_method",
    normalize=True,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## variable associations

# COMMAND ----------

logging.basicConfig(level=logging.DEBUG)

# COMMAND ----------

df_assoc_course = pdp.eda.compute_pairwise_associations(
    df_course_valid,
    exclude_cols=[
        "student_guid",
        "institution_id",
        "student_age",
        "gender",
        "race",
        "ethnicity",
    ],
)
df_assoc_course

# COMMAND ----------

course_assocs = (
    df_assoc_course.stack(future_stack=True)
    .reset_index(name="assoc")
    .rename(columns={"level_0": "col1", "level_1": "col2"})
    .sort_values(by="assoc", ascending=False)
    .loc[lambda df: df["col1"].ne(df["col2"]) & df["assoc"].notna(), :]
)
print(course_assocs.head(100).to_string())
print("...")

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

df_assoc_cohort = pdp.eda.compute_pairwise_associations(
    df_cohort_valid,
    exclude_cols=[
        "student_guid",
        "institution_id",
        "student_age",
        "gender",
        "race",
        "ethnicity",
    ],
)
df_assoc_cohort.head()

# COMMAND ----------

cohort_assocs = (
    df_assoc_cohort.stack(future_stack=True)
    .reset_index(name="assoc")
    .rename(columns={"level_0": "col1", "level_1": "col2"})
    .sort_values(by="assoc", ascending=False)
    .loc[lambda df: df["col1"].ne(df["col2"]) & df["assoc"].notna(), :]
)
print(cohort_assocs.head(100).to_string())
print("...")

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 12))
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
# MAGIC # Wrap-up

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC - [ ] Add school-specific data schemas and/or preprocessing functions into the appropriate directory in the [`student-success-intervention` repository](https://github.com/datakind/student-success-intervention)
# MAGIC - ...
