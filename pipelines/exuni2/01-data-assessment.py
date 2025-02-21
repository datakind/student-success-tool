# Databricks notebook source
# MAGIC %md
# MAGIC - general reference: https://docs.databricks.com/en/getting-started/ingest-insert-additional-data.html
# MAGIC - data source: https://help.studentclearinghouse.org/pdp/article-categories/6-analysis-readyfiles

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %pip install "pandas ~= 2.2.0"
# MAGIC %pip install "pandera ~= 0.20.0"
# MAGIC %pip install "pyarrow ~= 16.0"
# MAGIC %pip install "matplotlib ~= 3.9"
# MAGIC %pip install "missingno ~= 0.5.0"
# MAGIC %pip install "numpy ~= 1.26.0"
# MAGIC %pip install "seaborn ~= 0.13.0"
# MAGIC %pip install git+https://github.com/datakind/student-success-tool.git@develop

# COMMAND ----------

try:
    dbutils.library.restartPython()
except NameError:
    pass

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
import sys
from IPython.display import display

import missingno as msno
import pandas as pd
import seaborn as sns
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib
import shutil
from student_success_tool.analysis import pdp
import logging
from pyspark.sql import SparkSession
import missingno as msno

if "../" not in sys.path:
    sys.path.insert(1, "../")

from analysis.helpers import (
    create_numeric_summary_table,
    create_categorical_summary_table,
)
if not os.path.exists("tmp"):
    os.mkdir("tmp")
shutil.copyfile("../../custom_style.mplstyle", "tmp/custom_style.mplstyle")
plt.style.use("tmp/custom_style.mplstyle")
mplstyle_colors = [
    color_dict["color"] for color_dict in list(matplotlib.rcParams["axes.prop_cycle"])
]

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger

try:
    spark = SparkSession.builder.getOrCreate()
except Exception:
    logging.warning("unable to create spark session; are you in a Databricks runtime?")
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC # load raw data

# COMMAND ----------

catalog = "sst_dev"
bronze_schema = "bronze"
bronze_volume = "bronze_file_volume"
path_volume = os.path.join("/Volumes", catalog, bronze_schema, bronze_volume)
path_table = f"{catalog}.{bronze_schema}"
print(f"{path_table=}")
print(f"{path_volume=}")

# COMMAND ----------

dfs = {
    "cohort": "cohort",
    "course": "course"
}
df_cohort = spark.read.table(f"{catalog}.{bronze_schema}.{dfs['cohort']}").toPandas().replace("UK", np.nan).replace({None: np.nan}).replace("NA", np.nan).replace("Unknown", np.nan).replace(-1, np.nan)
df_course = spark.read.table(f"{catalog}.{bronze_schema}.{dfs['course']}").toPandas().replace("UK", np.nan).replace({None, np.nan}).replace("NA", np.nan).replace("Unknown", np.nan).replace(-1, np.nan)

# COMMAND ----------

df_cohort.head()

# COMMAND ----------

df_cohort.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # explore data

# COMMAND ----------

# MAGIC %md
# MAGIC ## overview

# COMMAND ----------

datasets = {
    "cohort": {
        "data": df_cohort,
        "unique_cols": ["StudentGUID"],
        "key_groups": [
            {"cols": "Cohort", "sort": False},
            {"cols": "EnrollmentType"},
            {"cols": ["StudentAge", "Gender"], "sort": False},
            {"cols": ["Race", "Ethnicity"], "sort": False},
        ],
    },
    "course": {
        "data": df_course,
        "unique_cols": [
            "StudentGUID",
            "AcademicYear",
            "AcademicTerm",
            "CoursePrefix",
            "CourseNumber",
            "SectionID",
        ],
        "key_groups": [
            {"cols": "CourseName", "num_to_display": 10},
            {"cols": "AcademicYear", "sort": False},
            {"cols": "AcademicTerm", "sort": False},
            {
                "cols": "EnrollmentRecordatOtherInstitutionsSTATEs",
                "num_to_display": 10,
            },
        ],
    },
}

# COMMAND ----------

for dname, dmeta in datasets.items():
    print(f"# {dname.upper()} DATASET")
    data = dmeta["data"]
    num_rows, num_cols = data.shape
    print(f"\nnum_rows = {num_rows}\nnum_cols = {num_cols}")

    if unique_cols := dmeta.get("unique_cols"):
        print("\n## unique identifiers")
        print(f"\ncolumns: {unique_cols}")
        uniques_result = pdp.eda.assess_unique_values(data, unique_cols)
        for key, val in uniques_result.items():
            print(f"{key} = {val}")

    print("\n## summary statistics")
    with pd.option_context("display.max_columns", 20, "display.width", 250):
        print("\n### categorical/string columns")
        print(pdp.eda.compute_summary_stats(data, include=['category', 'object']))
        print("\n### numeric/datetime columns")
        print(pdp.eda.compute_summary_stats(data, include=["number", "datetime"]))
        if key_groups := dmeta.get("key_groups"):
            print("\n### key groups")
            for key_group in key_groups:
                result = pdp.eda.compute_group_counts_pcts(
                    data, key_group["cols"], sort=key_group.get("sort", True)
                )
                num_groups = result.shape[0]
                if num_to_display := key_group.get("num_to_display"):
                    result = result.head(num_to_display)
                print(f"\n{result}")
                if num_to_display and num_to_display < num_groups:
                    print("...")

# COMMAND ----------

# MAGIC %md
# MAGIC Cohort
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

n_tabs = len(datasets.keys()) * 3
sub_tab = [widgets.Output() for i in range(n_tabs)]
tab = widgets.Tab(sub_tab)

i = 0
for key, data_dict in datasets.items():
    data = eval(f'df_{key}')
    unique_keys = data_dict.get("unique keys")

    if (unique_keys is not None) and (
        data[unique_keys].drop_duplicates().shape[0] != data.shape[0]
    ):
        print(f"Data is not unique by {unique_keys}. Try again!")

    tab.set_title(i, f"{key} dataset shape")
    with sub_tab[i]:
        print(data.shape)

    drop_cols = data_dict.get("drop columns")
    if drop_cols is not None:
        locals()[key] = data.drop(columns=drop_cols)
        data = eval(key)

    # classifying column types
    numeric_cols = data.select_dtypes("number").columns.values.tolist()
    date_cols = data.select_dtypes("datetime").columns.values.tolist()
    object_cols = data.select_dtypes("object").columns.values.tolist()
    assert (len(numeric_cols) + len(object_cols) + len(date_cols)) == data.shape[
        1
    ], "Error: missing column data types in categorizations."

    i += 1
    palette_i = 0
    tab.set_title(i, f"{key} numeric columns")
    with sub_tab[i]:
        num_date_cols = numeric_cols + date_cols
        print(f"Summary stats and plots of numeric columns in {key} data:")
        num_summary_df = create_numeric_summary_table(data)
        print(num_summary_df)
        for num_col in num_date_cols:
            if not data[num_col].isnull().all():
                plt.hist(data[num_col], color="teal")
                plt.ylabel(f"Number of {key} rows")
                plt.xlabel(num_col)
                plt.title(f"Frequency of {num_col}")
                plt.tight_layout()
                plt.show()
                palette_i += 1
                if palette_i == (len(mplstyle_colors) - 1):
                    palette_i = 0

    i += 1
    palette_i = 0
    tab.set_title(i, f"{key} categorical columns")
    with sub_tab[i]:
        print(f"Summary stats and plots of categorical columns in {key} data:")
        cat_summary_df = create_categorical_summary_table(data)
        print(cat_summary_df)
        for categorical_col in object_cols:
            frequencies = data[categorical_col].value_counts(dropna=True)
            print(frequencies)
            if frequencies.shape[0] <= 10:
                frequencies = pd.DataFrame(frequencies)
                frequencies = frequencies.reset_index()
                plt.bar(
                    x=frequencies[categorical_col],
                    height=frequencies["count"],
                    color="teal",
                )
                plt.title(f"Frequency of {categorical_col}")
                plt.xlabel(categorical_col)
                plt.ylabel(f"Number of {key} rows")
                plt.xticks(rotation=50)
                plt.tight_layout()
                plt.show()
                palette_i += 1
                if palette_i == (len(mplstyle_colors) - 1):
                    palette_i = 0
            print("\n")
    i += 1
display(tab)

# COMMAND ----------

frequencies = df_cohort["EnrollmentIntensityFirstTerm"].value_counts(dropna=True)
print(frequencies)
if frequencies.shape[0] <= 10:
    frequencies = pd.DataFrame(frequencies)
    frequencies = frequencies.reset_index()
display(frequencies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## nulls

# COMMAND ----------

_ = msno.matrix(df_course, sparkline=False)

# COMMAND ----------

_ = msno.matrix(df_cohort, labels=True)

# COMMAND ----------

df_cohort_not_missing = df_cohort.dropna(axis = 1, how='all') 
_ = msno.matrix(df_cohort_not_missing, labels=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## distributions

# COMMAND ----------

df_cohort["Cohort"] = pd.Categorical(df_cohort['Cohort'], ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"])
fig = sns.histplot(data=df_cohort, x="Cohort", hue="CohortTerm", multiple="dodge", shrink = 0.8)
fig.set_ylabel("Number of Students")
fig.set_title("Number of Students by Cohort and Term")
plt.show()

# COMMAND ----------

(
    sns.histplot(
        df_cohort,
        x="EnrollmentType",
        hue="EnrollmentIntensityFirstTerm",
        multiple="dodge",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students")
)

# COMMAND ----------

fig = sns.histplot(data=df_cohort, x="Cohort", hue="EnrollmentIntensityFirstTerm", multiple="dodge", shrink = 0.8)
fig.set_ylabel("Number of Students")
fig.set_title("Enrollment Intensity by Cohort")
plt.show()

# COMMAND ----------

# same as plot above, only in cross-tab form
100 * pdp.eda.compute_crosstabs(
    df_cohort,
    "EnrollmentType",
    "EnrollmentIntensityFirstTerm",
    normalize=True,
)

# COMMAND ----------

(
    sns.histplot(
        df_course.sort_values(by="AcademicYear"),
        x="AcademicYear",
        hue="AcademicTerm",
        multiple="dodge",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Course Enrollments")
)

# COMMAND ----------

print(
    "number of distinct courses:",
    df_course["CoursePrefix"].str.cat(df_course["CourseNumber"]).nunique(),
)

# COMMAND ----------

df_course['Course'] = df_course["CoursePrefix"].str.cat(df_course["CourseNumber"])
course_counts = df_course.groupby("Cohort")["Course"].nunique().reset_index()
fig = sns.barplot(data=course_counts, x="Cohort", y="Course", color = "teal")
fig.set_ylabel("Number of Courses")
fig.set_title("Number of Courses by Cohort")
display(fig)

# COMMAND ----------

course_counts_delivery = df_course.groupby(["Cohort", "DeliveryMethod"])["Course"].nunique().reset_index()
# display(course_counts_delivery)
fig = sns.barplot(data = course_counts_delivery, x="Cohort", y = "Course", hue="DeliveryMethod")
fig.set_ylabel("Number of Courses")
fig.set_title("Number of Courses by Cohort and Delivery Method")
display(fig)

# COMMAND ----------

students_in_course = df_course.groupby("Course")["StudentGUID"].nunique().reset_index()
students_in_course = students_in_course[students_in_course["StudentGUID"] >= 5]
display(students_in_course.sort_values("StudentGUID", ascending=False))
fig = sns.histplot(students_in_course, x="StudentGUID")
fig.set_xlabel("Number of Students in Courses")
fig.set_ylabel("Number of Courses")
fig.set_title("Number of Students in Courses by Course")
display(fig)

# COMMAND ----------

ax = sns.histplot(
    pd.merge(
        df_course.groupby("StudentGUID")
        .size()
        .rename("num_courses_enrolled")
        .reset_index(drop=False),
        df_cohort[["StudentGUID", "CredentialTypeSoughtYear1"]],
        on="StudentGUID",
        how="inner",
    ),
    x="num_courses_enrolled",
    hue="CredentialTypeSoughtYear1",
    binwidth=1,
    edgecolor="white",
)
ax.set(xlabel="Number of courses enrolled (total)", ylabel="Number of students")
sns.move_legend(ax, loc="upper left", bbox_to_anchor=(0.2, 1))

# COMMAND ----------

100 * df_cohort["credential_type_sought_year_1"].value_counts(normalize=True)

# COMMAND ----------

sns.jointplot(
    df_course.groupby("StudentGUID").agg(
        {"NumberofCreditsAttempted": "sum", "NumberofCreditsEarned": "sum"}
    ),
    x="NumberofCreditsAttempted",
    y="NumberofCreditsEarned",
    kind="hex",
    joint_kws={"bins": "log"},
    marginal_kws={"edgecolor": "white"},
    ratio=4,
    xlim=(0, 140),
    ylim=(0, 140),
)

# COMMAND ----------

(
    df_course.groupby("StudentGUID")
    .agg({"NumberofCreditsAttempted": "sum", "NumberofCreditsEarned": "sum"})
    .assign(
        num_credits_lost=lambda df: df["NumberofCreditsAttempted"]
        - df["NumberofCreditsEarned"]
    )
    .loc[:, "num_credits_lost"]
    .eq(0.0)
).value_counts()

# COMMAND ----------

sns.histplot(
    (
        df_course.groupby("StudentGUID")
        .agg({"NumberofCreditsAttempted": "sum", "NumberofCreditsEarned": "sum"})
        .assign(
            num_credits_lost=lambda df: df["NumberofCreditsAttempted"]
            - df["NumberofCreditsEarned"]
        )
    ),
    x="num_credits_lost",
    binwidth=1,
    edgecolor="white",
).set(xlabel="Number of credits lost (total)", ylabel="Number of students")

# COMMAND ----------

(
    sns.barplot(
        df_cohort,
        x="GPAGroupYear1",
        y="Cohort",
        estimator="mean",
        hue="EnrollmentType",
        edgecolor="white",
    ).set(xlabel="GPA (Year 1)")
)

# COMMAND ----------

print(
    df_cohort["TimetoCredential"]
    .isna()
    .rename("'Time to Credential' is null")
    .value_counts(dropna=False)
)
(
    sns.histplot(
        df_cohort,
        x="TimetoCredential",
        hue="EnrollmentType",
        # hue="enrollment_intensity_firstterm",
        binwidth=1,
        edgecolor="white",
    ).set(ylabel="Number of Students")
)

# COMMAND ----------

ax = sns.histplot(
    df_course.sort_values(by="Grade"),
    x="Grade",
    hue="DeliveryMethod",
    multiple="dodge",
    shrink=0.75,
    edgecolor="white",
)
ax.set(xlabel="Number of Course Enrollments")
ax.legend(title="Delivery Method", labels=["Face-to-face", "Online", "Hybrid"])

# COMMAND ----------

print("grade percentage totals per delivery method")
print(
    100
    * pdp.eda.compute_crosstabs(
        df_course, "Grade", "DeliveryMethod", normalize="columns"
    )
)
ax = sns.countplot(
    df_course,
    x="Grade",
    hue="DeliveryMethod",
    edgecolor="white",
    linewidth=0.5,
)
ax.set(xlabel="Number of Courses")
ax.legend(title="Delivery Method", labels=["Face-to-face", "Online", "Hybrid"])

# COMMAND ----------

print(
    "overall withdrawal rate:",
    100 * df_course["Grade"].eq("W").value_counts(normalize=True),
)

# COMMAND ----------

(
    df_course.loc[df_course["Grade"].isin(["0", "1", "2", "3", "4"]), "Grade"]
    .astype("string")
    .astype("Int64")
    .describe()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - **Q:** Does null "time to credential" mean the student never graduated / earned a credential?

# COMMAND ----------

# MAGIC %md
# MAGIC ## transform + merge datasets

# COMMAND ----------

pd.merge(
    df_course["StudentGUID"],
    df_cohort["StudentGUID"],
    on="StudentGUID",
    how="outer",
    indicator=True,
)["_merge"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### takeaways
# MAGIC
# MAGIC - All students in course dataset are also in cohort dataset, and vice-versa. Excellent!

# COMMAND ----------

# MAGIC %md
# MAGIC ## outcome variable

# COMMAND ----------

# MAGIC %md
# MAGIC For **ENROLLMENT TYPE TBD** students who earned at least 30 credits:
# MAGIC - enrolled full-time and earned credential within 3 years
# MAGIC - enrolled part-time and earned credential within 6 years
# MAGIC
# MAGIC What about "transfers out" to another institution by end of second year? What about earning credentials at that other institution?

# COMMAND ----------

df_cohort['NumberofCreditsEarned'] = df_cohort["NumberofCreditsEarnedYear1"] + df_cohort["NumberofCreditsEarnedYear2"] + df_cohort["NumberofCreditsEarnedYear3"] + df_cohort["NumberofCreditsEarnedYear4"]
df_outcome = df_cohort.loc[
    df_cohort["EnrollmentType"].eq("First-Time") & df_cohort["NumberofCreditsEarned"].ge(30.0), :
].assign(
    outcome=lambda df: (
        (
            df_cohort["EnrollmentIntensityFirstTerm"].eq("Full-Time")
            & df_cohort["TimetoCredential"].le(3.0)
        )
        | (
            df_cohort["EnrollmentIntensityFirstTerm"].eq("Part-Time")
            & df_cohort["TimetoCredential"].le(6.0)
        )
    )
)

# COMMAND ----------

display(df_cohort.filter(like='NumberofCreditsEarned'))

# COMMAND ----------

df_cohort['NumberofCreditsEarnedYear1'].describe()

# COMMAND ----------

df_cohort["EnrollmentIntensityFirstTerm"].value_counts()

# COMMAND ----------

pdp.eda.compute_group_counts_pcts(df_outcome, "outcome")

# COMMAND ----------

(
    sns.histplot(
        df_outcome,
        y="enrollment_intensity_first_term",
        hue="outcome",
        multiple="stack",
        shrink=0.75,
        edgecolor="white",
    ).set(xlabel="Number of Students", ylabel="Enrollment Intensity First Term")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### bias checks

# COMMAND ----------

100 * pdp.eda.compute_crosstabs(df_outcome, "student_age", "outcome", normalize="index")

# COMMAND ----------

100 * pdp.eda.compute_crosstabs(df_outcome, "gender", "outcome", normalize="index")

# COMMAND ----------

100 * pdp.eda.compute_crosstabs(df_outcome, "race", "outcome", normalize="index")

# COMMAND ----------

100 * pdp.eda.compute_crosstabs(
    df_outcome, "pell_status_first_year", "outcome", normalize="index"
)

# COMMAND ----------

(
    df_outcome.select_dtypes(include=["number", "datetime"])
    .corrwith(df_outcome["outcome"], method="spearman")
    .dropna()
)

# COMMAND ----------

sns.stripplot(
    df_outcome,
    x="time_to_credential",
    y="student_age",
    hue="outcome",
    dodge=True,
    jitter=0.2,
    alpha=0.5,
).set(xlabel="Time to Credential (years)", ylabel="Student Age", xlim=(0, 15))

# COMMAND ----------

sns.stripplot(
    df_outcome,
    x="time_to_credential",
    y="gender",
    hue="outcome",
    dodge=True,
    jitter=0.2,
    alpha=0.5,
).set(xlabel="Time to Credential (years)", ylabel="Student Gender", xlim=(0, 15))

# COMMAND ----------

ax = sns.stripplot(
    df_outcome,
    x="time_to_credential",
    y="race",
    hue="outcome",
    dodge=True,
    jitter=0.2,
    alpha=0.5,
)
ax.set(xlabel="Time to Credential (years)", ylabel="Student Race", xlim=(0, 15))

# COMMAND ----------

ax = sns.stripplot(
    df_outcome,
    x="time_to_credential",
    y="pell_status_first_year",
    hue="outcome",
    dodge=True,
    jitter=0.2,
    alpha=0.5,
)
ax.set(
    xlabel="Time to Credential (years)", ylabel="Pell Status (first year)", xlim=(0, 15)
)
