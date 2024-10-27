# Databricks notebook source
# %pip install faker pandas


# COMMAND ----------

num_records = 100  # how many records to generate

columns = [
    "StuNum",
    "First_Enrollment_Date",
    "FirstGenFlag",
    "AgeGroup",
    "Gender",
    "Race",
    "Ethnicity",
    "IncarceratedFlag",
    "DualEnrollFlag",
    "ABEFlag",
    "Zip",
    "DisabilityFlag",
    "HighSchoolStatus",
    "HighSchoolGradDate",
    "HighSchoolGPA",
    "StudentAthlete",
    "EarnedAward",
    "AwardsEarned",
]

# Define choices for each column
stu_num_range = range(10000, 99999)  # Unique student numbers
first_gen_choices = ["Y", "N", "U"]
age_group_choices = [
    "Under 18",
    "18-19",
    "20-24",
    "25-29",
    "30-39",
    "40-49",
    "50-64",
    "65+",
]
gender_choices = ["Female", "Male", "Not Specified"]
race_choices = ["Non-Hispanic", "Hispanic"]
ethnicity_choices = [
    "Two or more races",
    "Caucasian/White",
    "Unknown",
    "Black or African American",
    "American Indian or Alaska Native",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "Nonresident Alien",
]
incarcerated_choices = [0, 1]
dual_enroll_choices = [0, 1]
abe_choices = [0, 1]
disability_choices = ["N", "Y"]
high_school_status_choices = [
    "TraditionalHighSchoolDiploma",
    "No High School Record",
    "GED",
]


# COMMAND ----------

import random
from datetime import datetime

import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()


# Function to generate random dates
def random_date(start, end):
    return fake.date_between(start_date=start, end_date=end).strftime("%m/%d/%Y")


# Define the parameters for the dataset
data = []

for _ in range(num_records):
    stu_num = fake.unique.random_int(min=10000, max=99999)  # Unique student number
    first_enrollment_date = random_date(datetime(2000, 1, 1), datetime.now())
    first_gen_flag = random.choice(first_gen_choices)
    age_group = random.choice(age_group_choices)
    gender = random.choice(gender_choices)
    race = random.choice(race_choices)
    ethnicity = random.choice(ethnicity_choices)
    incarcerated_flag = random.choice(incarcerated_choices)
    dual_enroll_flag = random.choice(dual_enroll_choices)
    abe_flag = random.choice(abe_choices)
    zip_code = fake.zipcode()  # Generate arbitrary ZIP code
    disability_flag = random.choice(disability_choices)
    high_school_status = random.choice(high_school_status_choices)
    high_school_grad_date = random.choice(
        [random_date(datetime(1990, 1, 1), datetime.now()), None]
    )
    high_school_gpa = random.choice(
        [None] * 80 + [round(random.uniform(0, 4), 2) for _ in range(20)]
    )  # 80% nulls
    student_athlete = random.choice(["Y", "N"])
    earned_award = random.choice([None] * 80 + ["Y" for _ in range(20)])  # 80% nulls
    awards_earned = random.choice(
        [None] * 80 + [random.randint(1, 10) for _ in range(20)]
    )  # 80% nulls

    data.append(
        [
            stu_num,
            first_enrollment_date,
            first_gen_flag,
            age_group,
            gender,
            race,
            ethnicity,
            incarcerated_flag,
            dual_enroll_flag,
            abe_flag,
            zip_code,
            disability_flag,
            high_school_status,
            high_school_grad_date,
            high_school_gpa,
            student_athlete,
            earned_award,
            awards_earned,
        ]
    )
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("DataKind_School1_StudentFile.csv", index=False)


# COMMAND ----------

fake = Faker()

# Define choices for each column
semesters = [
    "2018FA",
    "2018SP",
    "2019FA",
    "2019SP",
    "2020FA",
    "2020SP",
    "2021FA",
    "2021SP",
]
enrollment_types = ["PT", "FT"]
degree_types = [
    "Associate of Science",
    "Associate of Applied Science",
    "Associate of Arts",
    "Associate of Business",
    "Associate of General Studies",
    "Certificate",
    "Non-Academic",
]
pell_recipient_choices = [None, "Y"]
majors = [
    "Custodian Studies Certificate",
    "Construction Technician",
    "Agriculture Associate of Arts",
    "Business Administration",
    "Nursing",
    "Computer Science",
    "Graphic Design",
]
department = [None]  # All NULL

columns = [
    "Student_ID",
    "Semester",
    "Enrollment_Type",
    "Degree_Type_Pursued",
    "Intent_To_Transfer_Flag",
    "Pell_Recipient",
    "Major",
    "Department",
    "Cumulative_GPA",
    "Semester_GPA",
    "Number_Of_Courses_Enrolled",
    "Number_Of_Credits_Attempted",
    "Number_Of_Credits_Failed",
    "Number_Of_Credits_Earned",
    "Course_Pass_Rate",
    "Online_Course_Rate",
]


# Function to enforce GPA and credit rules
def generate_valid_enrollment_data():
    number_of_courses = random.randint(0, 10)
    if number_of_courses == 0:
        number_of_credits_attempted = 0
        number_of_credits_failed = 0
        number_of_credits_earned = 0
    else:
        number_of_credits_attempted = random.randint(
            number_of_courses, 30
        )  # Ensure at least as many credits as courses
        number_of_credits_failed = random.randint(0, number_of_credits_attempted)
        number_of_credits_earned = (
            number_of_credits_attempted - number_of_credits_failed
        )

    cumulative_gpa = round(random.uniform(0, 4), 2)
    semester_gpa = round(random.uniform(0, 4), 2)
    course_pass_rate = round(random.uniform(0, 1), 2)
    online_course_rate = round(random.uniform(0, 1), 2)

    return [
        number_of_courses,
        number_of_credits_attempted,
        number_of_credits_failed,
        number_of_credits_earned,
        cumulative_gpa,
        semester_gpa,
        course_pass_rate,
        online_course_rate,
    ]


data = []

for _ in range(num_records):
    student_id = fake.unique.random_int(min=10000, max=99999)  # Unique student ID
    semester = random.choice(semesters)
    enrollment_type = random.choice(enrollment_types)
    degree_type = random.choice(degree_types)
    intent_to_transfer_flag = random.choice(
        ["Y", "N"]
    )  # Not specified in requirements but added for completeness
    pell_recipient = random.choice(pell_recipient_choices)
    major = random.choice(majors)
    dept = random.choice(department)  # All NULL

    # Generate valid enrollment data
    (
        number_of_courses,
        number_of_credits_attempted,
        number_of_credits_failed,
        number_of_credits_earned,
        cumulative_gpa,
        semester_gpa,
        course_pass_rate,
        online_course_rate,
    ) = generate_valid_enrollment_data()

    data.append(
        [
            student_id,
            semester,
            enrollment_type,
            degree_type,
            intent_to_transfer_flag,
            pell_recipient,
            major,
            dept,
            cumulative_gpa,
            semester_gpa,
            number_of_courses,
            number_of_credits_attempted,
            number_of_credits_failed,
            number_of_credits_earned,
            course_pass_rate,
            online_course_rate,
        ]
    )

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("DataKind_School1_SemesterFile.csv", index=False)


# COMMAND ----------

fake = Faker()

columns = [
    "Student_ID",
    "Semester",
    "Course_Prefix",
    "Course_Number",
    "Course_Type",
    "Pass/Fail_Flag",
    "Grade",
    "Prerequisite_Course_Flag",
    "Online_Course_Flag",
    "Number_Of_Credits_Attempted",
    "Number_Of_Credits_Earned",
    "Modality",
    "Weeks",
]

# Define choices for each column
course_prefixes = ["BIO", "COM", "ENG", "MAT", "HIS", "CHE", "PHY", "ART", "MUS", "PSY"]
course_types = ["Undergraduate"]  # All courses are undergraduate
pass_fail_choices = ["Y", "N"]
prerequisite_flag_choices = ["Y", None]
online_flag_choices = [1, 0]
modalities = [
    "Online",
    "Face to Face",
    "Online/Synch",
    "Hybrid - Online/iTV",
    "Hybrid - F2F/Online",
    "Independent Study",
    "Practicum",
]

# Load previously generated student IDs and semesters
semester_df = pd.read_csv("DataKind_School1_SemesterFile.csv")
student_ids = semester_df["Student_ID"].unique()
semesters = semester_df["Semester"].unique()

# Define the parameters for the dataset
data = []

# For each student and semester, generate multiple courses
for student_id in student_ids:
    # Get the semesters for this student
    student_semesters = semester_df[semester_df["Student_ID"] == student_id]

    for _, semester_row in student_semesters.iterrows():
        semester = semester_row["Semester"]
        number_of_courses = semester_row["Number_Of_Courses_Enrolled"]

        for _ in range(number_of_courses):
            course_prefix = random.choice(course_prefixes)
            course_number = random.randint(0, 299)
            course_type = random.choice(course_types)
            pass_fail_flag = random.choice(pass_fail_choices)
            grade = random.choice(
                ["A", "B", "C", "D", "F", None]
            )  # Include None for null grades
            prerequisite_flag = random.choice(prerequisite_flag_choices)
            online_course_flag = random.choice(online_flag_choices)
            number_of_credits_attempted = random.randint(
                1, 4
            )  # Assuming credits between 1 and 4
            number_of_credits_earned = random.choice(
                [number_of_credits_attempted, None]
            )  # Could be null
            modality = random.choice(modalities)
            weeks = random.randint(1, 16)  # Typical course duration in weeks

            data.append(
                [
                    student_id,
                    semester,
                    course_prefix,
                    course_number,
                    course_type,
                    pass_fail_flag,
                    grade,
                    prerequisite_flag,
                    online_course_flag,
                    number_of_credits_attempted,
                    number_of_credits_earned,
                    modality,
                    weeks,
                ]
            )

df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("DataKind_School1_CourseFile.csv", index=False)
