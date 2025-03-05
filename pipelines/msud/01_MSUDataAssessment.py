# Databricks notebook source
# MAGIC %md
# MAGIC - data source: https://help.studentclearinghouse.org/pdp/article-categories/6-analysis-readyfiles

# COMMAND ----------

# MAGIC %md
# MAGIC # setup

# COMMAND ----------

#%pip install "pandas ~= 2.2.0"
#%pip install "pandera ~= 0.20.0"
#%pip install "pyarrow ~= 16.0"
#%pip install "matplotlib ~= 3.9"
#%pip install "missingno ~= 0.5.0"
#%pip install "numpy ~= 1.26.0"
#%pip install "seaborn ~= 0.13.0"

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# install dependencies, most/all of which should come through our 1st-party SST package
# NOTE: it's okay to use 'develop' or a feature branch while developing this nb
# but when it's finished, it's best to pin to a specific version of the package
%pip install "student-success-tool == 0.1.0"
%pip install "git+https://github.com/datakind/student-success-tool.git@v0.1.0"

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
import sys
import schemas
import missingno as msno
import pandas as pd
import seaborn as sb
import numpy as np
import ipywidgets as widgets

from student_success_tool.analysis import pdp
import matplotlib
import matplotlib.pyplot as plt
import shutil
shutil.copyfile("../../custom_style.mplstyle", "/tmp/custom_style.mplstyle")
plt.style.use("/tmp/custom_style.mplstyle")
mplstyle_colors = [
    color_dict["color"] for color_dict in list(matplotlib.rcParams["axes.prop_cycle"])
]

# HACK: let's insert our 1st-party code into PATH
# TODO: we should probably install it proper into env

from databricks.connect import DatabricksSession
spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC # load raw data

# COMMAND ----------

catalog = "sst_dev"
inst_id = "metro_state_uni_denver"
read_schema = f"{inst_id}_bronze"
read_volume = f"{read_schema}_file_volume"
read_path_volume = os.path.join("/Volumes", catalog, read_schema, read_volume)
read_path_table = f"{catalog}.{read_schema}"
print(f"{read_path_table=}")
print(f"{read_path_volume=}")
write_schema = f"{inst_id}_silver"

# COMMAND ----------

spark.sql(f"LIST '{read_path_volume}'").collect()

# COMMAND ----------

files_dict = {'course': {'file_name': 'MSU_COURSE_LEVEL_AR_DEID_20240801150800.csv'},
              'course_20241218': {'file_name': 'MSU_COURSE_LEVEL_AR_DEID_20241218_121826.csv'}, 
              'course_20241029': {'file_name': 'MSU_denver_COURSE_LEVEL_AR_DEID_20241029000414_oct30_2024.csv'},
              'cohort': {'file_name': 'MSU_STUDENT_SEMESTER_LEVEL_AR_DEIDENTIFIED_20240801151000.csv'},
              'cohort_20241029': {'file_name': 'MSU_denver_AR_DEIDENTIFIED_20241029000400_oct30_2024.csv'},
              'cohort_20241218': {'file_name': 'MSU_AR_DEIDENTIFIED_20241218_121800.csv'}}

# COMMAND ----------

fpath_course = os.path.join(read_path_volume, files_dict['course']['file_name'])
df_course_raw = pdp.dataio.read_raw_pdp_course_data_from_file(fpath_course, schema=None)
print(df_course_raw.shape)
df_course_raw.head()

# COMMAND ----------

fpath_course_20241218 = os.path.join(read_path_volume, files_dict['course_20241218']['file_name'])
df_course_raw_20241218 = pdp.dataio.read_raw_pdp_course_data_from_file(fpath_course_20241218, schema=None)
fpath_course_20241029 = os.path.join(read_path_volume, files_dict['course_20241029']['file_name'])
df_course_raw_20241029 = pdp.dataio.read_raw_pdp_course_data_from_file(fpath_course_20241029, schema=None)

print(df_course_raw_20241218.shape)
display(df_course_raw_20241218.head())
print(df_course_raw_20241029.shape)
display(df_course_raw_20241029.head())


# COMMAND ----------

baseline_tuples = set(map(tuple, df_course_raw.values))

# COMMAND ----------

# Filter df_course_raw_20241218: keep rows that are NOT in the baseline (all 34 columns must match)
df_course_raw_20241218_unique = df_course_raw_20241218[~df_course_raw_20241218.apply(tuple, axis=1).isin(baseline_tuples)]

# COMMAND ----------

print(df_course_raw.student_guid.nunique())
print(df_course_raw_20241218_unique.student_guid.nunique())

# COMMAND ----------

unique_ids = set(df_course_raw.student_guid.unique()).intersection(df_course_raw_20241218_unique.student_guid.unique())
print(len(unique_ids))
print(unique_ids)

# COMMAND ----------

import random
rand_unique_id = random.choice(list(unique_ids))
df_course_raw[df_course_raw.student_guid == rand_unique_id]

# COMMAND ----------

df_course_raw_20241218_unique[df_course_raw_20241218_unique.student_guid == rand_unique_id]

# COMMAND ----------

# MAGIC %md
# MAGIC The file sent on 2024-12-18 is filled with 2024 course data which is not needed

# COMMAND ----------

# Filter df_course_raw_20241218: keep rows that are NOT in the baseline (all 34 columns must match)
df_course_raw_20241029_unique = df_course_raw_20241029[~df_course_raw_20241029.apply(tuple, axis=1).isin(baseline_tuples)]

# COMMAND ----------

print(df_course_raw.student_guid.nunique())
print(df_course_raw_20241029_unique.student_guid.nunique())

# COMMAND ----------

unique_ids = set(df_course_raw.student_guid.unique()).intersection(df_course_raw_20241029_unique.student_guid.unique())
print(len(unique_ids))
print(unique_ids)

# COMMAND ----------

import random
rand_unique_id = random.choice(list(unique_ids))
df_course_raw[df_course_raw.student_guid == rand_unique_id]

# COMMAND ----------

df_course_raw_20241029_unique[df_course_raw_20241029_unique.student_guid == rand_unique_id]

# COMMAND ----------

# MAGIC %md
# MAGIC The file sent on 2024-10-29 is filled with NaNs and does not add much information to the existing data

# COMMAND ----------

# re-read file without any column typing or validation to make sure there aren't numeric Grades
pd.read_csv(fpath_course)['Grade'].value_counts(dropna = False)

# COMMAND ----------

fpath_cohort = os.path.join(read_path_volume, files_dict['cohort']['file_name'])
df_cohort_raw = pdp.dataio.read_raw_pdp_cohort_data_from_file(fpath_cohort, schema=None)
print(df_cohort_raw.shape)
df_cohort_raw.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # fix primary key duplicates
# MAGIC - TODO: consider if we want to add any of these into automatic data validation or just utilities to use

# COMMAND ----------

# MAGIC %md
# MAGIC ## takeaways about duplicates
# MAGIC Course data has duplicates because of:
# MAGIC - multiple Cohort Terms per student. We overwrite the Cohort Term by using the Cohort file as the source of truth.
# MAGIC - multiple Grades per student/year/term/course/section. When the duplicate grade is a complete grade vs. incomplete one, we'll keep the complete grade. When the grade is a passing grade versus a worse grade, we'll keep the passing grade.

# COMMAND ----------

# check the differences between the Cohort Term in the course file vs. the cohort file
cohort_term_overlap = df_course_raw[['student_guid','cohort', 'cohort_term']].drop_duplicates().merge(df_cohort_raw[['student_guid', 'cohort', 'cohort_term']], on = 'student_guid', suffixes = ['_course','_cohort'])
cohort_term_overlap[['cohort_term_course','cohort_term_cohort']].value_counts()

# COMMAND ----------

df_course_corrected_cohort_term = df_course_raw.drop(columns = 'cohort_term').drop_duplicates().merge(df_cohort_raw[['student_guid','cohort_term']], on = 'student_guid')
print(f'Dropped {df_course_raw.shape[0] - df_course_corrected_cohort_term.shape[0]} duplicate course rows due to multiple Cohort Terms per student')

# COMMAND ----------

# There are more duplicated - check when these duplicate rows are happening
keys = ['student_guid','academic_year','academic_term','course_prefix','course_number','section_id']
grade_dupes = df_course_corrected_cohort_term[df_course_corrected_cohort_term.duplicated(keys, keep = False)]
# all of the grade duplicates happen in spring/summer of 2022
grade_dupes.groupby(['academic_year', 'academic_term']).agg({'student_guid': ['nunique','count']})

# COMMAND ----------

# what do the duplicate grade combinations look like? We want to prioritize the most "complete" grades first, so P > W, F > W, P > I, P > M. We'll also keep the better grade in the cases of [P, F], so P > F.
grade_dupes.sort_values('grade').groupby(keys, observed = True).agg({'grade': list}).reset_index()['grade'].value_counts()

# COMMAND ----------

original_n_null_grades = df_course_corrected_cohort_term.grade.isna().sum()
# sort grades by most "complete" and "better"
df_course_corrected_cohort_term['grade'] = pd.Categorical(df_course_corrected_cohort_term['grade'], categories = ['P', 'F', 'W', 'I', 'A', 'M', 'O'], ordered = True)
# make sure no grades got converted to null in the categorical conversion
assert original_n_null_grades == df_course_corrected_cohort_term['grade'].isna().sum()
df_course_deduped_grades = df_course_corrected_cohort_term.sort_values('grade').drop_duplicates(keys, keep = 'first')
assert df_course_deduped_grades.duplicated(keys).sum() == 0
print(f'Dropped {df_course_corrected_cohort_term.shape[0] - df_course_deduped_grades.shape[0]} duplicate course rows due to multiple grades per student per course')

# COMMAND ----------

# good, we expect that P and F are the grades remaining
df_course_deduped_grades.merge(grade_dupes[keys], on = keys)['grade'].value_counts()

# COMMAND ----------

df_course_deduped_grades.grade.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # validate raw data

# COMMAND ----------

df_course_deduped_grades.to_csv('/tmp/df_course_deduped_grades.csv', index=False)

# COMMAND ----------

df_course = pdp.dataio.read_raw_pdp_course_data_from_file('/tmp/df_course_deduped_grades.csv', schema=pdp.schemas.RawPDPCourseDataSchema, dttm_format="%Y-%m-%d")
df_course

# COMMAND ----------

df_cohort_raw

# COMMAND ----------

df_cohort_raw['enrollment_intensity_first_term'].unique()

# COMMAND ----------

df_cohort_raw[df_cohort_raw['enrollment_intensity_first_term'] == 'UNKNOWN']

# COMMAND ----------

df_cohort_raw['retention'] = df_cohort_raw["retention"].astype(int)
df_cohort_raw['persistence'] = df_cohort_raw["persistence"].astype(int)

# COMMAND ----------

print('Here are the unique values')
display(df_cohort_raw['first_gen'].unique())
print('Here are the null values')
print(df_cohort_raw['first_gen'].isnull().sum())
df_cohort_raw['first_gen'] = df_cohort_raw['first_gen'].astype('object')
print('Here are the rows with null values')
df_cohort_raw[df_cohort_raw['first_gen'].isnull()]

# COMMAND ----------

# There are only 13 null vals in first_gen, makes sense to fill with mode
df_cohort_raw['first_gen'].fillna(df_cohort_raw['first_gen'].mode()[0], inplace=True)
df_cohort_raw['first_gen'] = df_cohort_raw['first_gen'].apply(lambda x: x.strip())
print('Sanizing the first_gen column')
#df_cohort_raw['first_gen'] = df_cohort_raw['first_gen'].replace({'N': 0, 'Y': 1}).astype(int)
# Check the type of each element in the column
print(df_cohort_raw['first_gen'].apply(type).value_counts())

# COMMAND ----------

df_cohort_raw['credential_type_sought_year_1'] = df_cohort_raw['credential_type_sought_year_1'].apply(lambda x: x.strip())
df_cohort_raw['credential_type_sought_year_1'] = df_cohort_raw['credential_type_sought_year_1'].astype('object')
mapping = {
    'UNKNOWN': 'Missing',
    'Non Credential Program (Preparatory Coursework / Teach Certification)':
        'Non- Credential Program (Preparatory Coursework/Teach Certification)',
    'Less than 1-year certificate, less than Associates degree':
        'Less than one-year certificate, less than Associate degree',
    '1-2 year certificate, less than Associates degree':
        'One to two year certificate, less than Associate degree',
    'Non Credential Program (Preparatory Coursework / Teacher Certification)':
        'Non- Credential Program (Preparatory Coursework/Teach Certification)',
    '2-4 year certificate, less than Bachelors degree':
        "Two to four year certificate, less than Bachelor's degree"
}

df_cohort_raw['credential_type_sought_year_1'] = df_cohort_raw['credential_type_sought_year_1'].replace(mapping, regex=False)
df_cohort_raw['credential_type_sought_year_1'].unique()

# COMMAND ----------

df_cohort_raw['enrollment_intensity_first_term'] = df_cohort_raw['enrollment_intensity_first_term'].apply(lambda x: x.strip())
df_cohort_raw['enrollment_intensity_first_term'] = df_cohort_raw['enrollment_intensity_first_term'].astype('object')
df_cohort_raw['enrollment_intensity_first_term'].unique()
df_cohort_raw[df_cohort_raw['enrollment_intensity_first_term'] == 'UNKNOWN']
df_cohort_raw['enrollment_intensity_first_term'] = df_cohort_raw['enrollment_intensity_first_term'].str.replace('UNKNOWN', df_cohort_raw['enrollment_intensity_first_term'].mode()[0], regex=False)
print(df_cohort_raw['enrollment_intensity_first_term'].unique())

# COMMAND ----------

df_cohort_raw['first_gen'] = df_cohort_raw['first_gen'].astype('category')
df_cohort_raw['credential_type_sought_year_1'] = df_cohort_raw['credential_type_sought_year_1'].astype('category')
# Assuming 'FULL-TIME' as a default value, or you could use the mode (most frequent value)
#default_value = 'FULL-TIME'


# COMMAND ----------

print(df_cohort_raw['first_gen'].dtypes)
print(df_cohort_raw['credential_type_sought_year_1'].dtypes)
print(df_cohort_raw['enrollment_intensity_first_term'].dtypes)

# COMMAND ----------

#df_cohort_raw.to_csv('/tmp/df_cohort_exported.csv', index=False)

# COMMAND ----------

df_cohort_raw.first_year_to_bachelor_at_other_inst

# COMMAND ----------

df_cohort = schemas.RawPDPCohortDataSchema.validate(df_cohort_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC ### schema takeaways
# MAGIC
# MAGIC - A lot of the quirks for MSU Den are the same as the HACC quirks, as noted in the schema.
# MAGIC - Some new nuances, different from HACC, but nothing crucial to address here:
# MAGIC   - Credential Type Sought Year 1 was not validated for HACC, but I find slightly different categories for MSU Denver than the PDP documentation
# MAGIC   - The YearsTo fields have a maximum of 8, not 7.

# COMMAND ----------

# MAGIC %md
# MAGIC # assess course nulls

# COMMAND ----------

# sort first to make the null matrix potentially more meaningful
sorted_df_course = df_course.sort_values('course_begin_date')

# COMMAND ----------

_ = msno.matrix(sorted_df_course, sparkline=False)

# COMMAND ----------

# let's pretend that "unknown" categorical values are nulls
_ = msno.matrix(sorted_df_course.replace(to_replace=["UK", "Unknown", "UNKNOWN", "-1.0"], value=np.nan))

# COMMAND ----------

# MAGIC %md
# MAGIC ## patterns in course nulls
# MAGIC
# MAGIC TODO: We should automate filtering out this data - per PDP, null course indicators are a student's subsequent enrollment at another instituiton.

# COMMAND ----------

# Course identifiers are always null together (# of nulls = 3 or 0). 11% of rows affected by null course identifiers
sorted_df_course[['course_prefix','course_number','section_id']].isna().sum(axis = 1).value_counts().to_frame(name = 'count').assign(pct = lambda df: 100*df['count']/df['count'].sum())

# COMMAND ----------

# check if there are any patterns in when these nulls occur
possible_related_cols = ['cohort','academic_year','academic_term','enrolled_at_other_institution_s']

for possible_pattern_col in possible_related_cols:
    print(sorted_df_course.assign(null_course_prefix = lambda df: df['course_prefix'].isnull()).groupby(possible_pattern_col)['null_course_prefix'].mean())

# COMMAND ----------

# for the rows that have null course identifiers, what other columns are null?
sorted_df_course[pd.isnull(sorted_df_course['course_prefix'])].isna().mean()

# COMMAND ----------

# for the rows that have null course identifiers, what do the other columns look like?
for possible_pattern_col in possible_related_cols:
    print(sorted_df_course[pd.isnull(sorted_df_course['course_prefix'])][possible_pattern_col].value_counts(dropna=False, normalize = True))

# COMMAND ----------

df_course_valid = sorted_df_course[pd.notnull(sorted_df_course['course_prefix'])]

# COMMAND ----------

# MAGIC %md
# MAGIC ## takeaways about course nulls
# MAGIC - 11% of course records have a null course prefix
# MAGIC - Grade and Number of Credits are also null with these rows
# MAGIC - We filter out this subsequent enrollment data. In the future, we may choose to use this data as part of the outcome variable -- schools may consider a transfer to another institution (in addition to completion) a success.

# COMMAND ----------

# MAGIC %md
# MAGIC ## reassess course nulls with invalid rows removed

# COMMAND ----------

_ = msno.matrix(df_course_valid, sparkline=False)

# COMMAND ----------

course_nulls = df_course_valid.replace(to_replace=["UK", "Unknown", "UNKNOWN","-1.0"], value=np.nan).isna().mean().to_frame(name='pct_null').reset_index(names='colname')
pdp.dataio.write_data_to_delta_table(course_nulls, f"{catalog}.{write_schema}.course_nulls", spark)

# COMMAND ----------

# fields we will not use
course_nulls[course_nulls.pct_null == 1]['colname'].tolist()

# COMMAND ----------

# fields we will most likely not use
course_nulls[(0.25 < course_nulls.pct_null) & (course_nulls.pct_null < 1)]

# COMMAND ----------

# MAGIC %md
# MAGIC # assess cohort nulls

# COMMAND ----------

cohort_nulls = df_cohort.replace(to_replace=["UK", "Unknown","UNKNOWN","-1.0"], value=np.nan).isna().mean().to_frame(name='pct_null').reset_index(names='colname')
pdp.dataio.write_data_to_delta_table(cohort_nulls, f"{catalog}.{write_schema}.cohort_nulls", spark)

# COMMAND ----------

# fields we will not use
cohort_nulls[cohort_nulls.pct_null == 1]['colname'].tolist()

# COMMAND ----------

# fields we will most likely not use BESIDES the "Most Recent...", "First..." fields and "Time to Credential". These are null because of a high percentage of students that have not yet received a degree/certificate/credential.
cohort_nulls[(0.25 < cohort_nulls.pct_null) & (cohort_nulls.pct_null < 1)]

# COMMAND ----------

# Limit the visual to only the columns that have >0 nulls
sorted_df_cohort = df_cohort.sort_values('cohort')
has_null_cols = cohort_nulls[cohort_nulls.pct_null > 0].colname

_ = msno.matrix(sorted_df_cohort[has_null_cols], labels=True)

# COMMAND ----------

_ = msno.matrix(
    sorted_df_cohort[has_null_cols].replace(to_replace=["UK", "Unknown", "UNKNOWN", "-1.0"], value=np.nan), labels=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC # data insights

# COMMAND ----------

# MAGIC %md
# MAGIC ## explore categorical variables

# COMMAND ----------

def summarize_plot_categorical_column(dataset_name, colname, custom_color, write_schema, plot_dir = 'plots/'):
    frequencies = pdp.eda.compute_group_counts_pcts(eval(dataset_name), colname, dropna = False)
    frequencies.index = frequencies.index.astype(str).str.replace('nan', 'Blank') # handle Categorical NaNs in the plot
    print(frequencies.head(10))
    if frequencies.shape[0] > 10:
        print('...')
    if (observed_cat_frequencies := frequencies[frequencies['count'] > 0]).shape[0] > 1:
        plt.bar(
            x=observed_cat_frequencies.index,
            height=observed_cat_frequencies['count'],
            color=custom_color,
        )
        plt.title(f"Frequency of {colname}")
        plt.xlabel(colname)
        plt.ylabel(f"Number of {dataset_name} rows")
        plt.xticks(rotation=50)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{colname}.png"), bbox_inches="tight")
        plt.show()

dataset_options = widgets.Dropdown(options = ['df_cohort','df_course_valid'])
column_options = widgets.Dropdown()

def update(*args): # update the column names based on the choice of dataset
    selected_data = eval(dataset_options.value)
    column_options.options = selected_data.select_dtypes(include=["object","string","category"]).columns.values
dataset_options.observe(update)

interactive_plot = widgets.interact(summarize_plot_categorical_column, dataset_name=dataset_options, colname=column_options, custom_color = mplstyle_colors, write_schema = write_schema)
interactive_plot

# COMMAND ----------

# MAGIC %md
# MAGIC ## explore numeric variables

# COMMAND ----------

for dataset_name in ["df_cohort","df_course_valid"]:
    num_summary_df = pdp.eda.compute_summary_stats(eval(dataset_name), include=["number", "datetime","boolean"])
    print(num_summary_df)
    pdp.dataio.write_data_to_delta_table(num_summary_df.reset_index(names='colname'), f"{catalog}.{write_schema}.{dataset_name}_numeric_summary_stats", spark)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Questions for institution
# MAGIC - why are there Academic Years prior to Cohort years?
# MAGIC - No numeric grades at all?

# COMMAND ----------

# MAGIC %md
# MAGIC ## plots

# COMMAND ----------

# invalid course records? where academic year < cohort year
ax = sb.histplot(df_course_valid.assign(invalid_year = lambda x: x['academic_year'] < x['cohort']).\
                sort_values(['invalid_year',"cohort"], ascending = [False,True]),
            hue="invalid_year", x="cohort", multiple="stack", stat="count")
ax.set(ylabel = "Number of Course Enrollments")
ax.legend(labels=["Academic year < Cohort year", "Academic year >= Cohort year"])

# COMMAND ----------

# Perhaps this invalid data (Academic Year < Cohort) is due to dually enrolled students, but this data is not tracked -- all are Unknown
df_cohort['dual_and_summer_enrollment'].value_counts()

# COMMAND ----------

# number of students per cohort
sb.countplot(df_cohort.sort_values("cohort"), y="cohort").set(xlabel="Number of Students")

# COMMAND ----------

# average number of courses per student per term, over time
n_courses_per_student = df_course_valid.groupby(['student_guid','academic_year','academic_term'], observed = True).size().reset_index(name="n_courses")
sb.barplot(n_courses_per_student.astype({'academic_term': 'str'}).sort_values('academic_year'), x = "academic_year", hue = "academic_term", y = "n_courses", estimator="mean").set(ylabel = "Average number of courses per enrolled student")

# COMMAND ----------

# percent earned of attempted credits per academic year number
for year in [1,2,3,4]:
    year = str(year)
    df_cohort[f'pct_credits_earned_of_attempted_year_{year}'] = 100*df_cohort[f'number_of_credits_earned_year_{year}'] / df_cohort[f'number_of_credits_attempted_year_{year}']
pct_earned_of_attempted_df = df_cohort[['student_guid'] + df_cohort.columns[df_cohort.columns.str.startswith('pct')].tolist()]
pct_earned_of_attempted_df = pd.wide_to_long(pct_earned_of_attempted_df, stubnames = 'pct_credits_earned_of_attempted_year', i = 'student_guid', j = 'year').reset_index()
pct_earned_of_attempted_df.head()

# COMMAND ----------

# on average, students earn 78% of their attempted credits
pct_earned_of_attempted_df.mean()

# COMMAND ----------

sb.barplot(pct_earned_of_attempted_df, y = "pct_credits_earned_of_attempted_year", x = "year", estimator = "mean").set(ylabel = "Average % of Credits Earned out of Attempted", xlabel = "Academic Year Number", ylim = [0,100])

# COMMAND ----------

# 34% of students earn every credit they attempt
(
    df_course_valid.groupby("student_guid")
    .agg({"number_of_credits_attempted": "sum", "number_of_credits_earned": "sum"})
    .assign(
        num_credits_lost=lambda df: df["number_of_credits_attempted"]
        - df["number_of_credits_earned"]
    )
    .loc[:, "num_credits_lost"]
    .eq(0.0)
).value_counts(normalize = True)

# COMMAND ----------

# withdrawal rate over time, excluding possibly invalid academic years before 2016-17
df_course_valid['withdrawal'] = df_course_valid.grade.eq('W').astype(int) * 100
sb.barplot(df_course_valid[df_course_valid['academic_year'] >= '2016-17'].astype({'academic_term': 'str'}),
           x = "withdrawal", y = "academic_year", hue = "academic_term", estimator = "mean").set(xlabel = "% of Course Enrollments Withdrawn")

# COMMAND ----------

# MAGIC %md
# MAGIC ## check join conditions

# COMMAND ----------

# there are a couple students in the cohort file that are not in the deduplicated, validated course file. That makes sense!
join_check_df = pd.merge(
    df_course_valid[["student_guid","cohort","cohort_term"]],
    df_cohort[["student_guid","cohort","cohort_term"]],
    on=["student_guid","cohort","cohort_term"],
    how="outer",
    indicator=True,
)
join_check_df["_merge"].value_counts()

# COMMAND ----------

join_check_df[join_check_df._merge != 'both'] # there are some 2021-2022 and 2022-23 students not in the course file

# COMMAND ----------

# MAGIC %md
# MAGIC # write modified data

# COMMAND ----------

pdp.dataio.write_data_to_delta_table(df_course_valid, f"{catalog}.{write_schema}.course_valid", spark)
pdp.dataio.write_data_to_delta_table(df_cohort, f"{catalog}.{write_schema}.cohort_valid", spark)

# COMMAND ----------


