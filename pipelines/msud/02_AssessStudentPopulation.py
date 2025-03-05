# Databricks notebook source
# MAGIC %pip install "pandas ~= 2.2.0"
# MAGIC %pip install "pandera ~= 0.20.0"
# MAGIC %pip install "pyarrow ~= 16.0"
# MAGIC %pip install "matplotlib ~= 3.9"
# MAGIC %pip install "missingno ~= 0.5.0"
# MAGIC %pip install "numpy ~= 1.26.0"
# MAGIC %pip install "seaborn ~= 0.13.0"

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

import seaborn as sb
import matplotlib.pyplot as plt
from databricks.connect import DatabricksSession

import sys
sys.path.insert(1, '../')
from processing import silver_schema, catalog

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

silver_schema = silver_schema.format(inst_id = 'metro_state_uni_denver')

# COMMAND ----------

# MAGIC %md
# MAGIC # read data

# COMMAND ----------

df_course = spark.read.table(f"{catalog}.{silver_schema}.course_valid").toPandas() # noqa: F821
df_cohort = spark.read.table(f"{catalog}.{silver_schema}.cohort_valid").toPandas() # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC # assess transfer data
# MAGIC
# MAGIC Do we have any indicators of transfer credits? If we want to build a model off of transfer students, we need to know how many credits they are transferring in, in order to incorporate into the credit threshold filtering.
# MAGIC
# MAGIC Definition of "Enrolled at Other Institution(s)" from the [PDP schema](https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary/): "Whether or not the student was enrolled at another institution at the same time as this course or during this academic term (if a course record is not available)."
# MAGIC
# MAGIC Based on this definition, these records are not credits earned at other institutions -- these are courses that a student enrolled in at the partner institution, while concurrently enrolled at another institution. The flag is related to the academic term, not the course specifically.
# MAGIC
# MAGIC I don't see any other indicators, so TODO: we should filter out transfer students as part of our PDP pipeline
# MAGIC
# MAGIC For MSU Denver, this filters to 46% of their students (filters out the 54% of their students that are transfer students).

# COMMAND ----------

print(df_cohort['enrollment_type'].value_counts(normalize = True))
first_readmits = df_cohort[df_cohort.enrollment_type != 'TRANSFER-IN']
print(first_readmits['enrollment_type'].value_counts(normalize = True))

# COMMAND ----------

# MAGIC %md
# MAGIC # rough aggregation of course data
# MAGIC To get "earned" credits. In the future, this is where we would add more features.
# MAGIC
# MAGIC _Note on terminology_: MSU Denver (and maybe others) define "cumulative credits" = MSU Den credits + transfer credits. Since we are not looking at transfer students anymore, MSU Denver would call this "earned credits" = total credits earned to date at MSU Den

# COMMAND ----------

term_df = df_course.sort_values(['academic_year','academic_term']).groupby(['student_guid','academic_year','academic_term'], observed = True)['number_of_credits_earned'].sum().reset_index()
term_df['cumul_cred_earned'] = term_df.groupby('student_guid')['number_of_credits_earned'].cumsum()

# COMMAND ----------

term_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # merge & filter to a rough modeling dataset

# COMMAND ----------

possible_credit_thresholds = [60, 75, 90]

# COMMAND ----------

# MAGIC %md
# MAGIC We filter to Bachelor's-seeking students because:
# MAGIC 1. This is the majority of their student population
# MAGIC 2. They have chosen a credit threshold of 90. While we are still evaluating this threshold and may recommend an earlier checkpoint, it still likely does not make sense to include Associate's-seeking students, given that the credit threshold will probably be over the number of total credits Associate's degrees need, anyways (60).
# MAGIC
# MAGIC 96% of first time & readmit students are Bachelor's-seeking. If we had a term-level indicator, that would be better, but this is our best guess for now.

# COMMAND ----------

print(first_readmits['credential_type_sought_year_1'].value_counts(normalize = True))
first_readmits_bach = first_readmits[first_readmits.credential_type_sought_year_1 == "Bachelor's Degree"]
print(first_readmits_bach['credential_type_sought_year_1'].value_counts(normalize = True))

# COMMAND ----------

# MAGIC %md
# MAGIC How many students reach each possible credit threshold? MSU Denver expects this to be ~30% of their students.

# COMMAND ----------

joined_df = first_readmits_bach.merge(term_df, on = 'student_guid', how = 'inner')

for credit_threshold in possible_credit_thresholds:
    print(f'CREDIT THRESHOLD: {credit_threshold}')
    threshold_flag_col = f'threshold{credit_threshold}'
    joined_df[threshold_flag_col] = joined_df['cumul_cred_earned'] >= credit_threshold
    for student_group in ['enrollment_type','enrollment_intensity_first_term']:
        # did the student ever reach the credit threshold?
        print(joined_df.groupby(['student_guid',student_group]).agg({threshold_flag_col: 'max'}).\
            # what percent of students in the student group ever reached the credit threshold?
            groupby(student_group)[threshold_flag_col].mean())
    print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC Above, I'm still seeing a smaller number of students that reach 90 credits:
# MAGIC - 20% of FIRST-TIME students reach 90 credits, 9% of RE-ADMITS
# MAGIC - 23% of FULL-TIME students reach 90 credits, 7% of PART-TIME
# MAGIC Not 30% like MSU expects.
# MAGIC
# MAGIC We can also visualize this in a graph. At every credit threshold, how many students would be left? Note that this is overall, not disaggregated by enrollment type or intensity.

# COMMAND ----------

# Find the maximum number of cumulative credits earned, by student. Count how many students earned each number of credits
n_students_max_credits_df = joined_df.groupby('student_guid')['cumul_cred_earned'].max().reset_index().groupby('cumul_cred_earned').count().reset_index().sort_values('cumul_cred_earned')
# At each credit threshold, how many students did not meet the credit threshold. In other words, how many students would be excluded from the model?
n_students_max_credits_df['n_students_missing_threshold'] = n_students_max_credits_df['student_guid'].shift().cumsum().fillna(0)
# How many students would be left in the modeling dataset, if we filtered at that credit threshold?
n_students_max_credits_df['n_students_left'] = n_students_max_credits_df['student_guid'].sum() - n_students_max_credits_df['n_students_missing_threshold']
n_students_max_credits_df['pct_students_missing_threshold'] = n_students_max_credits_df['n_students_missing_threshold'] / n_students_max_credits_df['student_guid'].sum()
n_students_max_credits_df['pct_students_left'] = n_students_max_credits_df['n_students_left'] / n_students_max_credits_df['student_guid'].sum()

ax = sb.lineplot(data = n_students_max_credits_df, x = 'cumul_cred_earned', y = 'pct_students_left')
ax.set(xlim = [0,120], xlabel = 'Cumulative earned credits', ylabel = "% Bachelor's-seeking, FTIC or Readmit Students reaching threshold")
plt.axvline(90, 0, 1, linestyle = '--', color = '#009999')
plt.text(80, 0.7, '90-credit threshold', color = '#009999')

# COMMAND ----------

# MAGIC %md
# MAGIC # filter to records we have enough time into the future to evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC Our data ends in Fall '23. Given this, we need a full or part-time student to reach the credit threshold earlier enough in our data so that we have enough time into the future to evaluate our outcome by. This varies by credit threshold and full-time/part-time - we'll calculate those parameters below.

# COMMAND ----------

# what is our last term of data?
term_df[term_df['academic_year'] == term_df.academic_year.max()][['academic_year','academic_term']].value_counts()

# COMMAND ----------

ft_yrs_to_bach = 6
pt_yrs_to_bach = 8

tot_bach_credits = 120

credit_threshold_params = {credit_threshold: {'pct_left': 1 - (credit_threshold/tot_bach_credits)} for credit_threshold in possible_credit_thresholds}
for key, value in credit_threshold_params.items():
    credit_threshold_params[key]['FT_yrs_left'] = ft_yrs_to_bach * value['pct_left']
    credit_threshold_params[key]['PT_yrs_left'] = pt_yrs_to_bach * value['pct_left']
print(credit_threshold_params)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a term lookup table so we can easily identify records that are before the cutoff for each credit threshold & enrollment intensity

# COMMAND ----------

term_lkp_df = term_df[['academic_year','academic_term']].drop_duplicates().sort_values(['academic_year','academic_term'])
term_lkp_df['term_order'] = range(term_lkp_df.shape[0])
joined_df = joined_df.merge(term_lkp_df, on = ['academic_year','academic_term'], how = 'left')

# COMMAND ----------

term_lkp_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Find the latest term a student could reach the credit threshold, by credit threshold & enrollment intensity

# COMMAND ----------

last_term_rank = term_lkp_df.term_order.max()
print(last_term_rank)
terms_per_year = 3

for credit_threshold, params in credit_threshold_params.items():
    credit_threshold_params[credit_threshold]['FT_last_term_order'] = last_term_rank - params['FT_yrs_left'] * terms_per_year
    credit_threshold_params[credit_threshold]['PT_last_term_order'] = last_term_rank - params['PT_yrs_left'] * terms_per_year
credit_threshold_params

# COMMAND ----------

# MAGIC %md
# MAGIC For each credit threshold, flag if the student:
# MAGIC - reached the credit threshold
# MAGIC AND
# MAGIC - the term is early enough in time to measure the outcome, according to the student's enrollment intensity in their first term.

# COMMAND ----------

def flag_enough_data(enrl_intens, term_order, last_ft_term, last_pt_term):
    if (enrl_intens == 'FULL-TIME') and (term_order <= last_ft_term):
        return 1
    elif (enrl_intens == 'PART-TIME') and (term_order <= last_pt_term):
        return 1
    else:
        return 0

for credit_threshold, params in credit_threshold_params.items():
    print(f'CREDIT THRESHOLD : {credit_threshold}')
    
    enough_data_flag = f'enough_data{credit_threshold}'
    joined_df[enough_data_flag] = joined_df.apply(lambda x: flag_enough_data(x['enrollment_intensity_first_term'], x['term_order'], params['FT_last_term_order'], params['PT_last_term_order']), axis = 1)

    threshold_flag = f'threshold{credit_threshold}'
    print(joined_df[[enough_data_flag, threshold_flag]].value_counts())

    # by student that reached the threshold, did they ever have enough data?
    enough_data_df = joined_df.query(f'{threshold_flag} == 1').groupby(['enrollment_intensity_first_term','enrollment_type','student_guid'], observed = True)[enough_data_flag].max().reset_index()

    for student_group in ['enrollment_intensity_first_term','enrollment_type']:
        # by student_group, the percent of students and count of students that reached the threshold & had enough data
        student_group_enough_data_df = enough_data_df.groupby(student_group, observed = True).agg({enough_data_flag: 'mean', 'student_guid': 'count'}).reset_index()

        student_group_enough_data_df['n_students_left'] = student_group_enough_data_df[enough_data_flag] * student_group_enough_data_df['student_guid']
        print(student_group_enough_data_df)
        print('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC Adjusting the outcome variable from 12 years to 8 years for part-time students, there are still only 363 part-time students that reach 90 credits in the timeline to have enough data. 
# MAGIC
# MAGIC Looking at the other credit thresholds, there are still not enough part-time students to train a model on, and I don't want to recommend any earlier of a credit threshold, because our model accuracy will be poorer the further we get from completion.
# MAGIC
# MAGIC There are also not a lot of re-admit students, so the model would be dominated by full-time, first-time students.
# MAGIC
# MAGIC Some options:
# MAGIC - get more data into the past from NSC. Has MSU Denver already submitted this data & it would be an easy revision from NSC, or would they need to resubmit more data?
# MAGIC - move forward with first-time, full-time students only for this model
# MAGIC - move forward with first-time & readmit, full-time & part-time, knowing that the model patterns will be dominated by first-time/full-time students
# MAGIC
# MAGIC ## How much does this matter?
# MAGIC Check the outcome prevalence before and after filtering the data in this way. Does not filtering the data bias the outcome variable significantly?

# COMMAND ----------

# training data observations - the first term a student reaches the credit threshold
first_threshold_df = joined_df.query('threshold90 == 1').sort_values(['academic_year','academic_term']).groupby('student_guid').first().reset_index()

# outcome data - I'm choosing to use cumul credits >= 120 as the outcome, rather than Time to Credential or any of the other outcome data in the cohort file, but we should do a full analysis of this
credential_df = joined_df.query('cumul_cred_earned >= 120').sort_values(['academic_year','academic_term']).groupby('student_guid')['term_order'].first().reset_index()

# merge the training data and outcome data
prelim_modeling_df = first_threshold_df.merge(credential_df, on = ['student_guid'], how ='left', suffixes = ('_course','_credential'))

# Outcome: does the student graduate within 4.5 terms (1.5 years) -- TODO: we should ask about this conversion to terms from years. Effectively, this is 4 terms, since a student cannot be enrolled for half a term.
credit_threshold_params[90]['FT_yrs_left'] * terms_per_year

# COMMAND ----------

prelim_modeling_df['at risk'] = ~((prelim_modeling_df['term_order_credential'] - prelim_modeling_df['term_order_course']) <= 4)

# without filtering the data: prevalence = 56%
print(prelim_modeling_df['at risk'].value_counts(normalize = True))
print(prelim_modeling_df.shape[0]) # N = 4609

# with filtering the data, removes like 25% of our data, but prevalence skew is pretty large -- 47%
print(prelim_modeling_df.query('enough_data90 == 1')['at risk'].value_counts(normalize = True))
print(prelim_modeling_df.query('enough_data90 == 1').shape[0]) # N = 3303

# COMMAND ----------

modeling_df_enough_data = prelim_modeling_df.query('enough_data90 == 1')

# COMMAND ----------

# MAGIC %md
# MAGIC # compare first-time vs. readmit students
# MAGIC Let's also assess readmit students by the outcome prevalence. We'll calculate the outcome variable & then compare outcome prevalence.

# COMMAND ----------

# readmit students have a MUCH higher prevalence (84% vs 42%)
modeling_df_enough_data.groupby(['enrollment_type','enrollment_intensity_first_term']).agg({'at risk': 'mean', 'student_guid': 'count'})

# COMMAND ----------


