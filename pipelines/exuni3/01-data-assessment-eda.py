# Databricks notebook source
# MAGIC %md
# MAGIC # SST Data Assessment / EDA: exuni3
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
#%pip install git+https://github.com/datakind/student-success-tool.git@develop

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
# MAGIC ## unity catalog config

# COMMAND ----------

catalog = "sst_dev"

# configure where data is to be read from / written to
inst_name = "exuni3"  # TODO: fill in school's name in Unity Catalog
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

# TODO: fill in school's name; may not be same as in the schemas above
fpath_course = os.path.join(path_volume, "exuni3.csv")

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
df_course_raw = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course, schema=None, dttm_format="%Y%m%d.0"
)
print(f"rows x cols = {df_course_raw.shape}")
df_course_raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Quick checks:
# MAGIC - [ ] data exists where it should
# MAGIC - [ ] rows and columns are (roughly) as expected
# MAGIC - [ ] datetimes are correctly parsed (they're consistently, legibly formatted)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Managing duplicate rows with exuni3 course data
# MAGIC - There are 5 duplicate rows in the dataset. The following subset of columns are required to be unique together: `'student_guid', 'academic_year', 'academic_term', 'course_prefix', 'course_number', 'section_id'`. There are two students encompassing the 5 duplicate rows.
# MAGIC - This will be important to follow-up with exuni3 since the pairs of duplicate rows only differ in one column `'cohort_term'`. For example, a pair of duplicate rows may have "Summer" as a value in one row and "Spring" for the other row. It is unclear whether this is a data quality issue or this is an intentional choice.
# MAGIC - There are primarily two options for handling these rows. First, we could take the first or last row. Since these rows are relevant to only 2 students, we could also filter these students from the dataset.
# MAGIC - We will move forward with the latter of filtering the students from the dataset. 

# COMMAND ----------

# Find & filter duplicate rows
student_duplicates = df_course_raw[df_course_raw.duplicated(subset=['student_guid', 'academic_year', 'academic_term', 'course_prefix', 'course_number', 'section_id'])].student_guid.unique()
print(f"Found {len(student_duplicates)} duplicate rows")

# Create filtered dataframe
df_course_raw_ndupl = df_course_raw[~df_course_raw.student_guid.isin(student_duplicates)]

# Create new raw file/path with non-duplicate records for schema validation method below (requires raw file path)
df_course_raw_ndupl.to_csv(path_volume + '/exuni3_ndupl.csv', index=False)

# Update course path
fpath_course = os.path.join(path_volume, "exuni3_ndupl.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### exuni3 custom data format
# MAGIC - exuni3's dates do not follow default format given by our PDP template notebook `%Y%m%d.0`. 
# MAGIC - This is mostly due to the dashes in our date format, in which an example would be 2018-10-01.
# MAGIC - To help with this, we can define our own format below.

# COMMAND ----------

custom_exuni3_dttm_format = "%Y-%m-%d"

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
df_course = pdp.dataio.read_raw_pdp_course_data_from_file(
    fpath_course, schema=pdp.schemas.RawPDPCourseDataSchema, dttm_format=custom_exuni3_dttm_format
)
df_course

# COMMAND ----------

# MAGIC %md
# MAGIC If the above command works, and `df_course` is indeed a `pd.DataFrame` containing the validated + parsed PDP cohort dataset, then you're all set, and can skip ahead to the next section. If not, and this is instead a json blob of schema errors, then you'll need to iteratively develop school-specific overrides. There are existing examples you can refer to in the `student-success-intervention` repo.
# MAGIC
# MAGIC This will involve some ad-hoc exploratory work, depending on the schema errors. For example:
# MAGIC
# MAGIC ```python
# MAGIC # => any dupes? how to handle them??
# MAGIC >>> course_unique_cols = pdp.schemas.RawPDPCourseDataSchema.Config.unique
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
# MAGIC class RawSCHOOLCourseDataSchema(pdp.schemas.RawPDPCourseDataSchema):
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
# MAGIC df_course = pdp.dataio.read_raw_pdp_course_data_from_file(
# MAGIC     fpath_course,
# MAGIC     schema=RawSCHOOLCourseDataSchema,
# MAGIC     dttm_format="%Y%m%d.0",
# MAGIC     preprocess_func=ft.partial(
# MAGIC         course_data_preprocess_func,
# MAGIC         unique_cols=pdp.schemas.RawPDPCourseDataSchema.Config.unique
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


# TODO: fill in school's name; may not be same as in the schemas above
fpath_cohort = os.path.join(path_volume, "exuni3.csv")

# COMMAND ----------

# read without any schema validation, so we can look at the data "raw"
df_cohort_raw = pdp.dataio.read_raw_pdp_cohort_data_from_file(fpath_cohort, schema=None)
print(f"rows x cols = {df_cohort_raw.shape}")
df_cohort_raw.head()

# COMMAND ----------

# try to read data while validating with the "base" PDP schema
df_cohort = pdp.dataio.read_raw_pdp_cohort_data_from_file(
    fpath_cohort, schema=schemas.Rawexuni3CohortDataSchema
)
df_cohort

# COMMAND ----------

# MAGIC %md
# MAGIC Repeat the process from before, as needed: iteratively develop a school-specific cohort data schema and/or preprocessing function, then use them to produce the the validated/parsed data frame for analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## takeaways / questions
# MAGIC
# MAGIC - exuni3's cohort data appears to pass most "OK to be Null" checks, and there are many columns with nulls, which we will need to account for later.
# MAGIC - For now, it looks like "first_gen" needed to be overrided, as it doesn't appear to be defined with Y/N categories in template raw cohort file.
# MAGIC - There are no students who have credits attempted in year 2 or after, which is strange and doesn't appear to be consistent with students who are graduating at all.
# MAGIC - Overall, the data suggests very few students are graduating, which is likely a data quality issue, but we will move forward with building a retention model.  
# MAGIC - Math/English gateway column is 78% missing. From listening to previous recordings, this is likely because many students don't need the gateway.
# MAGIC
# MAGIC Questions:
# MAGIC - Why does the data suggest that few students are graduating? 
# MAGIC - Why are there no credits attempted/earned after year 1 for any students?
# MAGIC - About 20% of course-level data (number of credits attempted/earned, grades, and course begin/end dates) are missing. Why is this?

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
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ## read validated data
# MAGIC
# MAGIC (so you don't have to execute the validation process more than once)

# COMMAND ----------

# use base or school-specific schema, as needed
df_course = pdp.schemas.RawPDPCourseDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog}.{write_schema}.course_dataset_validated",
        spark_session=spark_session,
    )
)
df_course.shape

# COMMAND ----------

df_cohort = schemas.Rawexuni3CohortDataSchema(
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

dbutils.data.summarize(df_cohort, precise=True)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways / questions
# MAGIC
# MAGIC #### Missing Values (NA) Exploration Summary
# MAGIC - Prior to filtering academic years, 1482 out of 4117 students (around 35%) are missing course information. Once we filter academic years where more than 99% of data is missing, which is all years after 2023-24, then we have 1284 students (around 31%) are missing course information.
# MAGIC - About 20% of course-level data is completely missing. There are NA values across all columns in these rows. This is a combination of academic years that are completely missing and students who are missing data.
# MAGIC - It appears this is a known PDP issue and is due to these students transferring out of exuni3. This can be validated by the fact that all students have **Y** as a value for the  `enrolled_at_other_institution_s` column
# MAGIC - Data shows that many students transferred or dropped from Cohort 2019-2020. This is likely driven by COVID.
# MAGIC
# MAGIC ##### Course DF: Academic Years Missing
# MAGIC - 2023-24 has 99% of data missing, while 2024-25 and 2025-26 is missing 100%. We will likely need to filter these out.
# MAGIC - Even with filtering out missing academic years, there's still around 15% missing course-level data from other years.
# MAGIC
# MAGIC ##### Course DF: Students Missing Course-Level Data 
# MAGIC - There are 372 students out of 4117 (about 9%) that have 50% or more of their course level data missing.
# MAGIC - There are 76 students out of 4117 (about 2%) that have 80% or more of their course level data missing. 
# MAGIC - There isn't any particular trend with students who are missing 50% of course-level data. All cohorts have students who are missing data.
# MAGIC - When we look at students who are missing 80% of course-level data, the driver seems to be students from cohort 2019-2020. The academic years of these missing rows seem to be 2019-2020 and after.
# MAGIC
# MAGIC
# MAGIC ##### Cohort DF: Credits Attempted & Credits Earned
# MAGIC - The columns `number_of_credits_attempted_year_2`, `number_of_credits_attempted_year_3`, and `number_of_credits_attempted_year_4` as well as `number_of_credits_earned_year_2`, `number_of_credits_earned_year_3`, and `number_of_credits_earned_year_4` are all zeros.
# MAGIC - This is problematic because with the dataset as given, we cannot study graduation as a target variable. This limits us to studying retention for the time being.
# MAGIC
# MAGIC ##### Cohort DF: Time to Credential & Years to ... *fill in degree*
# MAGIC - These columns are related to years to associates and years to bachelors. These columns are missing values anywhere from 85% to 95%.
# MAGIC - Time to credential is missing 82% of data. 

# COMMAND ----------

# specific follow-ups, for example
print(df_course["grade"].value_counts(normalize=True, dropna=False))

# 2023 onwards is missing 99% or up to 100% of data. We will need to filter these out
missing_years = ['2023-24', '2024-25', '2025-26']
df_course_ay_filtered = df_course[~df_course.academic_year.isin(missing_years)]

students_missing_courselvl_all = df_course_ay_filtered[df_course_ay_filtered.grade.isna()].student_guid.unique() 
pct_missing_courselvl = 100 * (df_course_ay_filtered[df_course_ay_filtered.grade.isna()].student_guid.value_counts() 
                               / df_course_ay_filtered[df_course_ay_filtered.student_guid.isin(students_missing_courselvl_all)].student_guid.value_counts()) 
pct_missing_courselvl.sort_values(ascending=False).hist(bins=100)
plt.ylabel('# of Students')
plt.xlabel('% of Missing Course Information')
plt.title('exuni3 Students with Missing Course Information')
plt.show()

# Gather students with > 50% missing course information
students_50 = pct_missing_courselvl[pct_missing_courselvl > 50].index.to_list()

# Gather students with > 80% missing course information
students_80 = pct_missing_courselvl[pct_missing_courselvl > 80].index.to_list()

print(f"Number of students with missing course information: {len(students_missing_courselvl_all)}")
print(f"Number of students with > 50% missing course information: {len(students_50)}")
print(f"Number of students with > 80% missing course information: {len(students_80)}")

plt.figure()

df_course_ay_filtered[df_course_ay_filtered.student_guid.isin(students_80)].cohort.value_counts().plot.bar()
plt.ylabel('# of Students')
plt.xlabel('Cohort Year')
plt.title('Cohort Year of exuni3 Students with > 80% Missing Course Information')
plt.figure()

df_course_ay_filtered[df_course_ay_filtered.student_guid.isin(students_80)].academic_year.value_counts().plot.bar()
plt.ylabel('# of Students')
plt.xlabel('Academic Year')
plt.title('Academic Year of exuni3 Students with > 80% Missing Course Information')
plt.show()

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

# here's the null visualization for rows with students not enrolled at other institutions
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
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - There are 1482 students total that have transferred out of exuni3, and thus have at least one missing row of course-level information.
# MAGIC - We will need to filter out all data from 2023-24 onwards, along with all students who are missing course-level data.
# MAGIC
# MAGIC - The following columns are missing 100% of data:
# MAGIC     - completed_dev_math_y_1
# MAGIC     - completed_dev_english_y_1
# MAGIC     - foreign_language_completion
# MAGIC     - disability_status
# MAGIC     - military_status
# MAGIC     - incarcerated_status
# MAGIC     - special_program
# MAGIC     - reading_placement 
# MAGIC - For cohort data, there are many columns with missing values. Some of this may be due to a binary column that implies to have one value such as dual enrollment column. These have 'DE', 'SE', and 'DS' designations standing for dual enrollment, summer enrollment, and dual status. A NaN value here likely indicates that the student does not have any of these designations. Though we would need some follow-up here.
# MAGIC
# MAGIC
# MAGIC Questions:
# MAGIC Course -
# MAGIC 1. For `delivery_method`, there are values for F, O, H, and NaN. What does NaN imply here? _Proposal: Unknown_
# MAGIC
# MAGIC Cohort - 
# MAGIC 1. For `dual_and_summer_enrollment`, there are values for DE, SE, DS, and NaN. What does NaN mean here? _Proposal: Not DE, SE, or DS (no dual or summer enrollment)_
# MAGIC 2. For `attempted_dev_math_y_1` and `attempted_dev_english_y_1`, there are three values of Y, N, and NaN. What does NaN imply here? _Proposal: Unknown_
# MAGIC 3. For `gateway_english_grade_y_1` and `gateway_math_grade_y_1`, there are 7 values of P, F, M, W, O, I, and NaN. What are all of these values and what does NaN mean here? _Proposal: Unknown_
# MAGIC 4. For `program_of_study_term_1` and `program_of_study_year_1`, there are many numeric values (example: 513801.0). What do these mean? What does NaN mean here? 
# MAGIC 5. For `english_placement` and `math_placement`, there are 3 values of N, C, and NaN. What do N and C mean here? What does NaN mean?
# MAGIC 6. For `most_recent_last_enrollment_at_other_institution_locale`, `most_recent_last_enrollment_at_other_institution_carnegie`, `most_recent_last_enrollment_at_other_institution_state`, what does NaN mean here? _Proposal: Not applicable_
# MAGIC
# MAGIC

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
df.loc[df["_merge"] != "both", "student_guid"].unique().tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC #### takeaways / questions
# MAGIC
# MAGIC - There are two students that appear only in cohort data and not in the course dataset.
# MAGIC - This is because of our filter above since these two students had duplicate rows. 
# MAGIC Question for exuni3: Why do these students have duplicate rows?

# COMMAND ----------

# MAGIC %md
# MAGIC ### logical consistency

# COMMAND ----------

# check for "pre-cohort" courses
df = df[df.enrollment_type == 'FIRST-TIME']
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
plt.title('Pre-Cohort Courses for First-Time Students')

# COMMAND ----------

df_pre_cohort = df.loc[df["cohort"].gt(df["academic_year"]), :].assign(
    cohort_id=lambda df: df["cohort"].str.cat(df["cohort_term"], sep=" "),
    term_id=lambda df: df["academic_year"].str.cat(df["academic_term"], sep=" "),
)
df_pre_cohort[["student_guid", "cohort_id", "term_id", "enrollment_type", "cohort_term"]].cohort_id.value_counts(normalize=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### filter invalid rows(?)

# COMMAND ----------

df_course_filtered, df_cohort_filtered = preprocessing.filter_invalid(df_course, df_cohort)

# COMMAND ----------

# this is probably a filter you'll want to apply
# these courses known to be an issue w/ PDP data
df_course_valid = df_course.loc[df_course["course_number"].notna(), :]
df_course_valid

# COMMAND ----------

# MAGIC %md
# MAGIC ## plots and stats

# COMMAND ----------

# MAGIC %md
# MAGIC **Note:** You'll probably want to use the "valid" dataframes for most of these plots, but not necessarily for all. For simplicity, all these example plots will just use the base data w/o extra data validation filtering applied. It's your call!

# COMMAND ----------

# assign original dfs to filtered dfs for the sake of plotting/visualization below
df_course = df_course_filtered
df_cohort = df_cohort_filtered

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

ax = sb.histplot(
    pd.merge(
        df_course.groupby("student_guid")
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
    df_course.groupby("student_guid").agg(
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
# MAGIC ### takeaways / questions
# MAGIC
# MAGIC - A majority of students are pursuing their Bachelor's degree.
# MAGIC - Enrollment has dropped off since COVID, as it has been decreasing since 2020-21.
# MAGIC - Mean GPA is around 1.9, with a Median of 2.1.
# MAGIC - The _Number of Credits Earned_ by _Number of Credits Attempted_ plot is quite telling.
# MAGIC   - Most students who attempt around 35-40 credits earn most of their credits. 
# MAGIC   - There are MANY students who attempt anywhere from greater than 0 to 20 credits and earn 0 credits. It would be interesting to see how many of these students retain.

# COMMAND ----------

# MAGIC %md
# MAGIC ## variable associations

# COMMAND ----------

df_assoc_course = pdp.eda.compute_pairwise_associations(
    df_course,
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
    df_cohort,
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
# MAGIC # Wrap-up

# COMMAND ----------

# MAGIC %md
# MAGIC - [ ] Add school-specific data schemas and/or preprocessing functions into the appropriate directory in the [`student-success-intervention` repository](https://github.com/datakind/student-success-intervention)
