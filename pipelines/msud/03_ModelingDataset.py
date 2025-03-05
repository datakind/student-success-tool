# Databricks notebook source
# MAGIC %md
# MAGIC # setup

# COMMAND ----------

# install dependencies, most/all of which should come through our 1st-party SST package
# NOTE: it's okay to use 'develop' or a feature branch while developing this nb
# but when it's finished, it's best to pin to a specific version of the package
%pip install "student-success-tool == 0.1.0"
%pip install "git+https://github.com/datakind/student-success-tool.git@v0.1.0"

# COMMAND ----------

dbutils.library.restartPython() # noqa: F821

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import logging

import pandas as pd  # noqa: F401
import mlflow
import numpy as np
import sys
sys.path.insert(1, '../')

from student_success_tool.analysis import pdp
from student_success_tool.modeling import feature_selection

from processing import silver_schema, catalog
import yaml

import tomllib

with open("config.toml", "rb") as f:
    cfg = tomllib.load(f)
#typical example cfg.datasets[dataset_name].raw_course.file_path

logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # ignore databricks logger
mlflow.autolog(disable=True) # for the feature selection function

# COMMAND ----------

# MAGIC %md
# MAGIC # read data

# COMMAND ----------

silver_schema = silver_schema.format(inst_id = 'metro_state_uni_denver')

# COMMAND ----------

df_course = spark.read.table(f"{catalog}.{silver_schema}.course_valid").toPandas() # noqa: F821
df_cohort = spark.read.table(f"{catalog}.{silver_schema}.cohort_valid").toPandas() # noqa: F821

# COMMAND ----------

df_course.columns

# COMMAND ----------

df_course.course_number

# COMMAND ----------

# MAGIC %md
# MAGIC # transform and join datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ## preprocessing
# MAGIC Minor tweaks to the data prior to aggregating into a features dataset. Some of these are to match the format required by the pdp functionlity, others are modifications requested by MSU Denver

# COMMAND ----------

# MAGIC %md
# MAGIC ### exclude continuing education / non-credit bearing course data
# MAGIC MSU denver says that courses with <4 characters: "are continuing education / non-credit bearing and can be excluded from analysis."

# COMMAND ----------

# 1% of course data
nc_flag = df_course.course_number.str.len() < 4
(nc_flag).mean()

# COMMAND ----------

# 70% of these have 0 credits attempted -- seems mostly in line with their expectation, and negligible otherwise. Dropping!
print(df_course[(nc_flag)]['number_of_credits_attempted'].value_counts(normalize = True))

# COMMAND ----------

df_course_valid_courses = df_course[~nc_flag]
rows_dropped = df_course.shape[0] - df_course_valid_courses.shape[0]
print(f'Dropped {rows_dropped} ({100*rows_dropped/df_course.shape[0]}%) rows of course data that are continuing education/non-credit bearing.')

# COMMAND ----------

# MAGIC %md
# MAGIC ### also exclude course levels 5 & 6 per requested by MSU Denver, as these are graduate courses. study abroad (level 8) can be kept

# COMMAND ----------

# 4 digits or 3 digits + 1 capital letter or 2 digits + 2 capital letters
msu_den_course_level_pattern = r'^(?P<course_level>\d)\d{1,3}[A-Z]{,2}$'

# COMMAND ----------

df_course_valid_courses['course_level'] = pdp.features.course.course_level(df_course_valid_courses, col = 'course_number', pattern = msu_den_course_level_pattern)
print(df_course_valid_courses.groupby('course_level').size())

# COMMAND ----------

df_courses_no_grad = df_course_valid_courses[~df_course_valid_courses['course_level'].isin([5,6])].drop(columns = 'course_level')
dropped_rows = df_course_valid_courses.shape[0] - df_courses_no_grad.shape[0]
print(f'Dropped {dropped_rows} ({dropped_rows / df_course_valid_courses.shape[0]} %) graduate course rows.')

# COMMAND ----------

# MAGIC %md
# MAGIC ### create categorical columns
# MAGIC TODO: we could add this to the pipeline

# COMMAND ----------

df_courses_no_grad['grade'] = pd.Categorical(df_courses_no_grad['grade'])
df_courses_no_grad['academic_term'] = pd.Categorical(df_courses_no_grad['academic_term'], ["FALL", "WINTER", "SPRING", "SUMMER"], ordered = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ok now actually aggregate & merge the data

# COMMAND ----------

# standardize math course prefixes
df_courses_no_grad['course_prefix'] = np.where(df_courses_no_grad['course_prefix'] == 'MATHEMATICS_SCIENCE', 'MATHEMATICAL_SCIENCE', df_courses_no_grad['course_prefix'])

# standardize math course prefixes
df_courses_no_grad['course_prefix'] = np.where(df_courses_no_grad['course_prefix'] == 'MATHEMATICS_TEACHINGLEARNING', 'MATHEMATICS_TEACHING_LEARNING', df_courses_no_grad['course_prefix'])


# COMMAND ----------

df_courses_no_grad['course_prefix'].unique()

# COMMAND ----------

[s for s in df_courses_no_grad['course_prefix'].unique() if "MATH" in s]

# COMMAND ----------

key_course_ids = cfg['preprocessing']['features']['key_course_ids']

# COMMAND ----------

df_courses_no_grad.grade.unique()

# COMMAND ----------

df_courses_no_grad

# COMMAND ----------

df_student_terms = pdp.dataops.make_student_term_dataset(df_cohort = df_cohort,
                                                        df_course = df_courses_no_grad,
                                                        #institution_state = "CO",
                                                        course_level_pattern = msu_den_course_level_pattern,
                                                        key_course_ids = key_course_ids)

# COMMAND ----------

df_student_terms

# COMMAND ----------

df_student_terms.columns

# COMMAND ----------

[s for s in df_student_terms.columns if "frac_" in s]

# COMMAND ----------

# MAGIC %md
# MAGIC ### manual manipulation of features

# COMMAND ----------

# MSU Denver has not submitted numeric grades yet, so rename DFW features to just FW for now
import re
#df_student_terms = df_student_terms.rename(mapper= lambda x: x.replace('O|I|F|W', 'FW'), axis = 1)
#df_student_terms = df_student_terms.rename(mapper=lambda x: re.sub(r'(_F|_W)$', '_FW', x),axis=1)
# manipulate features for the key course IDs to make them more useful
df_student_terms['num_courses_key_general_studies'] = df_student_terms[['num_courses_course_id_' + course_id for course_id in key_course_ids]].sum(axis = 1)

for course_id in key_course_ids:
    df_student_terms[f'has_taken_{course_id}_this_term'] = df_student_terms[f'num_courses_course_id_{course_id}'] > 0
    # cumfrac includes the current term
    df_student_terms[f'has_taken_{course_id}_ever'] = df_student_terms[f'num_courses_course_id_{course_id}_cumfrac'] > 0

# drop the rest of the course ID cols
df_student_terms = df_student_terms.drop(columns = df_student_terms.filter(regex = "|".join(key_course_ids)).filter(regex='num_courses|frac').columns.tolist())

# COMMAND ----------

for i in df_student_terms.columns:
    print(i)

# COMMAND ----------

# MAGIC %md
# MAGIC ### manual addition of other features

# COMMAND ----------

df_student_terms_sorted = df_student_terms.sort_values(['student_guid','term_rank'])

# COMMAND ----------

# fraction of courses DFW in first term
df_first_term_F = (df_student_terms_sorted.groupby('student_guid')['frac_courses_course_grade_F'].first().reset_index().rename(columns={'frac_courses_course_grade_F': 'first_term_frac_courses_grade_F'}))

df_first_term_W = df_student_terms_sorted.groupby('student_guid')['frac_courses_course_grade_W'].first().reset_index().rename(columns={'frac_courses_course_grade_W': 'first_term_frac_courses_grade_W'})


# COMMAND ----------

# fraction of DFW courses last 1, 2, 3 semesters
for terms_ago in [1, 2, 3]:
    df_student_terms_sorted[f'frac_courses_grade_F_term_n-{terms_ago}'] = df_student_terms_sorted.groupby('student_guid')['frac_courses_course_grade_F'].shift(terms_ago)

# COMMAND ----------

# fraction of DFW courses last 1, 2, 3 semesters
for terms_ago in [1, 2, 3]:
    df_student_terms_sorted[f'frac_courses_grade_W_term_n-{terms_ago}'] = df_student_terms_sorted.groupby('student_guid')['frac_courses_course_grade_W'].shift(terms_ago)

# COMMAND ----------

# revised student terms dataframe
df_student_terms = df_student_terms_sorted.merge(df_first_term_F, on='student_guid').merge(df_first_term_W, on='student_guid')
assert df_student_terms.shape[0] == df_student_terms_sorted.shape[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### questions to ask MSU Denver about features:
# MAGIC - how to handle course numbers 'FY11', 'FY31', 'FY08', 'FY12' -- is there a course level? are these valid?

# COMMAND ----------

write_table_path = f"{catalog}.{silver_schema}.student_term_dataset_full"
pdp.dataio.write_data_to_delta_table(df_student_terms, write_table_path, spark) # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # filter
# MAGIC MSU Denver has decided to move forward with first-time & readmit, full-time & part-time, Bachelor's-seeking students, knowing that the model patterns will be dominated by first-time/full-time students. We will evaluate model performance on these subgroups afterwards.

# COMMAND ----------

MIN_NUM_CREDITS_CHECKIN = 90
MIN_NUM_CREDITS_TARGET = 120

# MSU Denver expects that full-time students should graduated in 6 years total, part-time students in 8 years
# Since our checkpoint is at 90 credits, we normalize the remaining time left. 90 credits/120 credits = 75% complete, so for full-time students, 75% of 6 years = 1.5 years left = 4 or 5 terms left, since 3 terms/year
# For part-time students, 75% of 8 years = 2 years left = 6 terms left
INTENSITY_VAL_TERMS_LEFTS = [('FULL-TIME', 4, "term"), ('PART-TIME', 6, "term")]

# COMMAND ----------

# TODO: I think it would make more sense for this function to be select_eligible_student_terms -- returning the term data that is eligible for each student.
df_eligible_students = pdp.targets.failure_to_earn_enough_credits_in_time_from_checkin.select_eligible_students(
    df_student_terms,
    student_criteria={
        "credential_type_sought_year_1": "Bachelor's Degree",
        "enrollment_type": ["FIRST-TIME",'RE-ADMIT'],
        "enrollment_intensity_first_term": ["FULL-TIME", "PART-TIME"],
    },
    min_num_credits_checkin=MIN_NUM_CREDITS_CHECKIN,
    intensity_time_lefts=INTENSITY_VAL_TERMS_LEFTS,
)
df_eligible_student_terms = pd.merge(
    df_student_terms, df_eligible_students, on=["student_guid"], how="inner"
)

# COMMAND ----------

df_eligible_student_terms.drop_duplicates("student_guid")["enrollment_intensity_first_term"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # create outcome variable

# COMMAND ----------

student_targets = pdp.targets.failure_to_earn_enough_credits_in_time_from_checkin.compute_target_variable(
    df_eligible_student_terms,
    min_num_credits_checkin=MIN_NUM_CREDITS_CHECKIN,
    min_num_credits_target=MIN_NUM_CREDITS_TARGET,
    intensity_time_lefts=INTENSITY_VAL_TERMS_LEFTS,
)
student_targets.value_counts()

# COMMAND ----------

student_targets.value_counts(normalize = True)

# COMMAND ----------

write_table_path = f"{catalog}.{silver_schema}.student_targets_dataset_full"
pdp.dataio.write_data_to_delta_table(
    student_targets.reset_index(drop=False), write_table_path, spark # noqa: F821
)

# COMMAND ----------

# MAGIC %md
# MAGIC # create modeling dataset

# COMMAND ----------

student_id_cols = ["institution_id", "student_guid"]
include_cols = [
    col for col in df_eligible_student_terms.columns
    if col not in student_id_cols
]
df_features = pdp.targets.shared.get_first_student_terms_at_num_credits_earned(
    df_eligible_student_terms,
    min_num_credits=MIN_NUM_CREDITS_CHECKIN,
    student_id_cols=student_id_cols,
    include_cols=include_cols,
)
df_labeled = pd.merge(df_features, student_targets, left_on='student_guid', right_index=True, how="inner").astype({"target": "bool"})
assert df_labeled.shape[0] == df_features.shape[0] == student_targets.shape[0]
df_labeled.shape

# COMMAND ----------

df_labeled['num_credits_earned_cumsum'].describe()

# COMMAND ----------

df_labeled['academic_year'].max()

# COMMAND ----------

df_labeled.target.value_counts(normalize = True)

# COMMAND ----------

# MAGIC %md
# MAGIC # correlations of all features before feature selection

# COMMAND ----------

df_corr = df_labeled.select_dtypes(include=['number','boolean','bool','datetime']).corr()['target'].sort_values()
df_corr = pd.DataFrame(df_corr.reset_index(name='target_correlation')).rename(columns={'index': 'column_name'})

# COMMAND ----------

df_corr.dropna().tail(15)

# COMMAND ----------

df_corr.dropna().head(15)

# COMMAND ----------

write_table_path = f"{catalog}.{silver_schema}.feature_correlations"
pdp.dataio.write_data_to_delta_table(
    df_corr, write_table_path, spark # noqa: F821
)

# COMMAND ----------

# MAGIC %md
# MAGIC # feature selection
# MAGIC ## first drop features not intended for modeling

# COMMAND ----------

df_labeled.columns

# COMMAND ----------

dropped_cols = cfg['preprocessing']['features']['dropped_cols']
df_labeled = df_labeled.drop(columns = dropped_cols +
                             # gateway is a highly null categorical (94%) so we should remove these numeric columns derived from that categorical
                             [col for col in df_labeled.columns.values if 'gateway' in col])

# COMMAND ----------

# MAGIC %md
# MAGIC # quick fixes & write data
# MAGIC TODO: add to the pipeline if generalizable & not intentional

# COMMAND ----------

df_labeled['student_program_of_study_area_term_1'] = df_labeled['student_program_of_study_area_term_1'].astype(str) # year_1 alternative should also be str
df_labeled['year_of_enrollment_at_cohort_inst'] = df_labeled['year_of_enrollment_at_cohort_inst'].clip(lower = 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## select features methodically

# COMMAND ----------

general_studies_flags = df_labeled.filter(regex='has_taken').columns.tolist()
fw_features = df_labeled.filter(regex='grade_FW').columns.tolist()

force_include_cols = cfg['modeling']['feature_selection']['force_include_cols'] + general_studies_flags + fw_features

selected_features_df = feature_selection.select_features(
    df_labeled,
    non_feature_cols=cfg['modeling']['feature_selection']['non_feature_cols'],
    force_include_cols=force_include_cols
).drop(columns = ['num_courses_enrolled_at_other_institution_s_Y', 'sections_num_students_passed', 'academic_year'], errors = 'ignore')

# COMMAND ----------

# Ensure selected_features is available from your configuration
selected_features = cfg['modeling']['feature_selection']['selected_features_baseline']

# Check if there are any features from selected_features not present in selected_features_df
missing_features = set(selected_features) - set(selected_features_df.columns.values)
if missing_features:
    print("Missing features from DataFrame:", missing_features)

# Assuming fw_features and general_studies_flags are defined earlier in your code
new_features = set(selected_features_df.columns.values) - set(selected_features) - set(fw_features + general_studies_flags)

print("New features added: general studies flags, FW features, and:", new_features)
print(f'Dataset shape: {selected_features_df.shape}')


# COMMAND ----------

[s for s in df_labeled.columns if 'student' in s]

# COMMAND ----------

# 'student_program_of_study_area_changed_first_year' is not 'student_program_of_study_changed_term_1_to_year_1'

# COMMAND ----------

len([s for s in selected_features_df.columns if 'has_taken' in s])

# COMMAND ----------

# MAGIC %md
# MAGIC Selected features cover the following categories:
# MAGIC - **Time**: academic_term, year_of_enrollment_at_cohort_inst, cohort_term, term_in_peak_covid, term_in_peak_covid_cumsum
# MAGIC - **Student type**: enrollment_type, enrollment_intensity_first_term
# MAGIC - **Enrollment**: term_is_while_student_enrolled_at_other_inst, term_is_while_student_enrolled_at_other_inst_cumsum, frac_courses_enrolled_at_other_institution_s_Y, cumfrac_fall_spring_terms_unenrolled
# MAGIC - **Subjects**: [student_]program_of_study_[area]_term_1, student_program_of_study[_area]_changed_first_year, course_subject_area_nunique
# MAGIC - **Courses**: delivery method O/H/F features
# MAGIC - **Section difficulty**: student_completion/pass_rate_above_sections_avg, course_level_std, course level features
# MAGIC - **Section size**: section_num_students_enrolled_std, section_num_students_enrolled_mean
# MAGIC - **Credits earned**: num_credits_earned_cummin, num_credits_earned_cumsum
# MAGIC
# MAGIC V2 selected features also cover:
# MAGIC - **Grade features**
# MAGIC - **Key general studies courses features**
# MAGIC - Earned credits / attempted credits
# MAGIC - Number of repeated courses
# MAGIC - Section pass rate

# COMMAND ----------

# MAGIC %md
# MAGIC # check remaining features

# COMMAND ----------

# minimal nulls, not imputing for now
selected_features_df.isna().mean().sort_values(ascending = False).head(20)

# COMMAND ----------

for col in selected_features_df.select_dtypes(include=['number', 'datetime']).columns:
    # Check for numeric data types
    if selected_features_df[col].dtype.kind in 'biufc':  # b=bool, i=int, u=unsigned int, f=float, c=complex
        assert selected_features_df[col].min() >= 0, f"Negative values found in {col}"
        if 'frac' in col:
            assert selected_features_df[col].max() <= 1, f"Values greater than 1 found in {col}"
        else:
            print(selected_features_df[col].describe())
    # Separate handling for datetime data
    elif selected_features_df[col].dtype.kind == 'M':  # M=datetime
        print(f"Column {col} is a datetime column with range from {selected_features_df[col].min()} to {selected_features_df[col].max()}")


# COMMAND ----------

for col in selected_features_df.select_dtypes(include=['object','category','boolean','bool']).columns.values:
    if col == 'student_guid':
        continue
    print(selected_features_df[col].value_counts(normalize = True, dropna = False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## prep for bias evaluation

# COMMAND ----------

np.random.seed(44133)
selected_features_df["split_col"] = np.random.choice(
    [
        "train",
        "test",
        "validate",
    ],  # Following split_col guidelines for a manual split from https://docs.databricks.com/en/machine-learning/automl/automl-data-preparation.html#split-data-into-train-validation-and-test-sets. Since AutoML does not preserve Student IDs in the training data, we need a way of knowing which rows were in which partition, for evaluating the model across student groups in the validation set.
    size=selected_features_df.shape[0],
    p=[0.6, 0.2, 0.2],  # matching default split strategy by AutoML
)
selected_features_df.split_col.value_counts(normalize = True)

# COMMAND ----------

write_table_path = f"{catalog}.{silver_schema}.training_data"
pdp.dataio.write_data_to_delta_table(selected_features_df, write_table_path, spark) # noqa: F821

# COMMAND ----------


