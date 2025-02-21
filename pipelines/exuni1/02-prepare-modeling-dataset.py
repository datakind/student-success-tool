# Databricks notebook source
# MAGIC %md
# MAGIC # SST Prepare Modeling Dataset: exuni1
# MAGIC
# MAGIC Second step in the process of transforming raw (PDP) data into actionable, data-driven insights for advisors: featurize the raw, validated data; configure and compute the target variable; perform feature selection; prepare train/test/validation splits; and inspect feature-target associations.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Data science product components (Confluence doc)](https://datakind.atlassian.net/wiki/spaces/TT/pages/237862913/Data+science+product+components+the+modeling+process)
# MAGIC - [Databricks Data preparation for classification](https://docs.databricks.com/en/machine-learning/automl/classification-data-prep.html)
# MAGIC - [Databricks runtimes release notes](https://docs.databricks.com/en/release-notes/runtime/index.html)
# MAGIC - [exuni1 website](https://www.exuni1.edu/)

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# MAGIC %sh python --version

# COMMAND ----------

# install dependencies, most of which should come through our 1st-party SST package
# %pip install "student-success-tool==0.1.0"
# %pip install git+https://github.com/datakind/student-success-tool.git@develop
# %pip install git+https://github.com/datakind/student-success-tool.git@pdp-fix-feature-name-mismatch-bug

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import functools as ft
import logging
import sys

import seaborn as sb
from databricks.connect import DatabricksSession
from student_success_tool import configs
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
from pipelines import exuni1  # noqa: F403

# COMMAND ----------

# MAGIC %md
# MAGIC ## project config

# COMMAND ----------

config = configs.load_config("./config.toml", schema=configs.PDPProjectConfig)
config

# COMMAND ----------

catalog = "sst_dev"

# configure where data is to be read from / written to
schema = f"{config.institution_name}_silver"
catalog_schema = f"{catalog}.{schema}"
print(f"{catalog_schema=}")

# COMMAND ----------

# MAGIC %md
# MAGIC # read validated data

# COMMAND ----------

df_course = pdp.schemas.RawPDPCourseDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog_schema}.course_dataset_validated", spark_session=spark_session
    )
)
print(f"rows x cols = {df_course.shape}")
df_course.head()

# COMMAND ----------

df_cohort = exuni1.schemas.Rawexuni1CohortDataSchema(
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

dict(config.prepare_modeling_dataset)

# COMMAND ----------

df_student_terms = pdp.dataops.make_student_term_dataset(
    df_cohort,
    df_course,
    min_passing_grade=config.prepare_modeling_dataset.min_passing_grade,
    min_num_credits_full_time=config.prepare_modeling_dataset.min_num_credits_full_time,
    course_level_pattern=config.prepare_modeling_dataset.course_level_pattern,
    key_course_subject_areas=config.prepare_modeling_dataset.key_course_subject_areas,
    key_course_ids=config.prepare_modeling_dataset.key_course_ids,
)
df_student_terms

# COMMAND ----------

# take a peek at featurized columns -- it's a lot
df_student_terms.columns.tolist()

# COMMAND ----------

# save student-term dataset in unity catalog (if needed)
# write_table_path = f"{catalog_schema}.student_term_dataset"
# pdp.dataio.write_data_to_delta_table(df_student_terms, write_table_path, spark_session)

# COMMAND ----------

# MAGIC %md
# MAGIC # filter students and compute target

# COMMAND ----------

df_student_terms = pdp.schemas.PDPStudentTermsDataSchema(
    pdp.dataio.read_data_from_delta_table(
        f"{catalog_schema}.student_term_dataset", spark_session=spark_session
    )
)
print(f"rows x cols = {df_student_terms.shape}")
df_student_terms.head()

# COMMAND ----------

student_id_cols = "student_guid"

# COMMAND ----------

df_labeled = pdp.targets.failure_to_retain.make_labeled_dataset(
    df_student_terms,
    student_criteria=config.prepare_modeling_dataset.target_student_criteria,
    student_id_cols=student_id_cols,
)
print(f"rows x cols = {df_labeled.shape}")
df_labeled.head()

# COMMAND ----------

df_labeled["target"].value_counts(normalize=True)

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

df_labeled = pdp.dataops.clean_up_labeled_dataset_cols_and_vals(df_labeled)
df_labeled.shape

# COMMAND ----------

# HACK: get rid of extra columns
# these *should* be null, except for "pre-cohort" students
# and let's not give them special advantages
drop_cols = (
    df_labeled.columns[df_labeled.columns.str.contains(r"_prev_term")].tolist()
    + df_labeled.columns[
        df_labeled.columns.str.contains(r"(?<!term_is_pre_cohort)_cum\w+$")
    ].tolist()
    + df_labeled.columns[df_labeled.columns.str.contains(r"^cum(?:num|frac)")].tolist()
)
print(f"dropping additional {len(drop_cols)} columns")
df_labeled = df_labeled.drop(columns=drop_cols)
df_labeled.shape

# COMMAND ----------

# save labeled dataset in unity catalog (if needed)
# write_table_path = f"{catalog_schema}.labeled_dataset"
# pdp.dataio.write_data_to_delta_table(df_labeled, write_table_path, spark_session=spark_session)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature selection + splits

# COMMAND ----------

import mlflow
from student_success_tool.modeling import utils
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
    force_include_cols=[
        "gpa_group_term_1",
        "frac_courses_grade_is_failing_or_withdrawal",
        "course_grade_numeric_mean",
        "frac_courses_grade_above_section_avg",
        "num_courses_in_program_of_study_area_term_1",
        "term_is_pre_cohort_cumsum",
        "frac_courses_course_subject_area_51",
        "frac_courses_course_id_englb_101",
        "frac_courses_course_id_englb_102",
    ],
    collinear_threshold=20.0,
)
df_labeled_selected

# COMMAND ----------

df_labeled_selected = df_labeled_selected.assign(
    split=ft.partial(utils.compute_dataset_splits, seed=2),
    sample_weight=ft.partial(
        utils.compute_sample_weights, target_col="target", class_weight="balanced"
    ),
)
df_labeled_selected.head()

# COMMAND ----------

df_labeled_selected["split"].value_counts(normalize=True)

# COMMAND ----------

# pdp.schemas.PDPLabeledDataSchema(df_labeled).dtypes
# df_labeled.shape

# COMMAND ----------

# # uncomment as needed
# write_table_path = f"{catalog_schema}.modeling_dataset"
# pdp.dataio.write_data_to_delta_table(df_labeled_selected, write_table_path, spark_session=spark_session)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature-target associations

# COMMAND ----------

df_corrs = df_labeled_selected.corrwith(
    df_labeled_selected["target"], method="spearman", numeric_only=True
).sort_values(ascending=False)
print(df_corrs.to_string())

# COMMAND ----------

target_assocs = pdp.eda.compute_pairwise_associations(
    df_labeled_selected,
    ref_col="target",
    exclude_cols=["student_guid", "split"],
)
target_assocs.sort_values(by="target")

# COMMAND ----------

# MAGIC %md
# MAGIC ### HACK: check avg grades by race and term

# COMMAND ----------

# MAGIC %md
# MAGIC For context: Our trained model moderately over-predicts African American students' non-retention.

# COMMAND ----------

foo = (
    df_student_terms.loc[
        df_student_terms["student_guid"].isin(df_labeled["student_guid"])
        & df_student_terms["term_rank"].le(
            df_student_terms["min_student_term_rank"] + 1
        ),
        [
            "student_guid",
            "race",
            "course_grade_numeric_mean",
            "min_student_term_rank",
            "term_rank",
        ],
    ]
    .assign(
        student_term_num=lambda df: df["term_rank"] - df["min_student_term_rank"] + 1
    )
    .drop(columns=["min_student_term_rank", "term_rank"])
)
foo

# COMMAND ----------

ag = sb.displot(
    foo.loc[foo["race"].isin(["WHITE", "HISPANIC", "BLACK OR AFRICAN AMERICAN"]), :],
    x="course_grade_numeric_mean",
    hue="race",
    multiple="dodge",
    col="student_term_num",
    bins=10,
)
ag.set(xlabel="avg. course grades for student-term", ylabel="number of students")

# COMMAND ----------

print(
    foo.loc[foo["race"].isin(["WHITE", "HISPANIC", "BLACK OR AFRICAN AMERICAN"]), :]
    .groupby(by=["race", "student_term_num"])
    .agg(avg_course_grade=("course_grade_numeric_mean", "mean"))
    .round(2)
)
