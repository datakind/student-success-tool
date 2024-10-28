# Databricks notebook source
import csv
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from pyspark.sql import SparkSession

# Create or get the existing Spark session
spark = SparkSession.builder.appName("YourAppName").getOrCreate()
# Replace 'your_table_name' with the actual name of the table you want to query
df = spark.sql("SELECT * FROM datakind.student_success_intervention.master_dataset")

# To show the first few rows of the DataFrame
df.show()


# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

df.columns

# COMMAND ----------

df

# COMMAND ----------

df_columns = [
    "UNIQUE_ID",
    "Term",
    "College ID",
    "Age",
    "CAS CAA English SUM 1",
    "CAS CAA Foreign Language 1 SUM 1",
    "CAS CAA Foreign Language 2 SUM 1",
    "CAS CAA Math SUM 1",
    "CAS CAA Performing Arts SUM 1",
    "CAS CAA Science SUM 1",
    "CAS CAA Social Studies SUM 1",
    "CAS College Admissions Avg (CAA) SUM 1",
    "CAS HS  Units English SUM 1",
    "CAS HS Units Foreign Language 1 SUM 1",
    "CAS HS Units Foreign Language 2 SUM 1",
    "CAS HS Units Math SUM 1",
    "CAS HS Units Performing Arts SUM 1",
    "CAS HS Units Science SUM 1",
    "CAS HS Units Social Studies SUM 1",
    "CAS HS Units Total SUM 1",
    "CAS Regents American History",
    "CAS Regents Biology",
    "CAS Regents Chemistry",
    "CAS Regents Common Core Alg1",
    "CAS Regents Common Core Alg2",
    "CAS Regents Common Core Geometry",
    "CAS Regents Earth Science",
    "CAS Regents English",
    "CAS Regents English New",
    "CAS Regents Geometry",
    "CAS Regents Global History",
    "CAS Regents Integ Alg",
    "CAS Regents Marine Science",
    "CAS Regents Math A",
    "CAS Regents Math B",
    "CAS Regents Math Sequential 1",
    "CAS Regents Math Sequential 2",
    "CAS Regents Math Sequential 3",
    "CAS Regents Physics",
    "CAS Regents Trigonometry",
    "CAS SAT Math",
    "CAS SAT Total",
    "CAS SAT Verbal",
    "CAS SAT Writing",
    "CS Total Income Amt SUM",
    "College Program Title",
    "Credits Cumulative Earned Local Perf",
    "Credits Cumulative Earned Local Show",
    "Credits Cumulative Earned Other Perf",
    "Credits Cumulative Earned Other Show",
    "Credits Cumulative Earned Total Perf",
    "Credits Cumulative Earned Total Show",
    "Credits Cumulative Earned Transfer Perf",
    "Credits Cumulative Earned Transfer Show",
    "Credits Semester Included in GPA Perf SUM",
    "Credits+Hours Semester All Courses Show SUM",
    "Credits+Hours Semester Completed Perf SUM",
    "Credits+Hours Semester Passed Perf SUM",
    "Cs Fed Efc Amt SUM",
    "Cs Fed Need Base Aid Amt SUM",
    "Cs Fed Unmet Need Amt SUM",
    "Degree Earned Level Desc",
    "Degree Pursued Level Desc",
    "Ethnicity Imputed Group 2 Desc",
    "Exit Math CUNY Test Final Score",
    "Exit Reading CUNY Test Final Score",
    "Exit Writing CUNY Test Final Score",
    "File Name",
    "Full Part Type Desc",
    "GPA Cumulative Perf",
    "GPA Cumulative Show",
    "GPA Semester Perf",
    "Gender Desc",
    "Headcount SUM",
    "Headcount SUM 1",
    "IR Total FA Awd Amt SUM",
    "Initial All SKAT Passed Desc",
    "Initial Math CUNY Test Final Score",
    "Initial Reading ACT Test Final Score",
    "Initial Writing CUNY Test Final Score",
    "Ir Fed Work Study Nb Awd Amt SUM",
    "Ir Student Loans Nb Awd Amt SUM",
    "Ir Total Grants Nb Awd Amt SUM",
    "Major 1 College Program Title",
    "Major 1 NYSED Award 1 Desc",
    "Major 1 NYSED Award 2 Desc",
    "Major 1 NYSED Program Title",
    "NYSED Award Desc",
    "NYSED Program Title",
    "New Student Desc",
    "PELL_AMT_AWD",
    "Pell Flag",
    "SEEK CD Desc",
    "Semester Enrolled Desc",
    "Semester Graduated Desc",
    "TAP Flag",
    "TAP_AMT_AWD",
    "Value Points Semester Perf SUM",
    "Year Enrolled",
    "Year Graduated",
]


# COMMAND ----------

# %pip install faker


# COMMAND ----------

# Create a Faker instance for generating fake data
fake = Faker()

# Define the number of rows for the dataset
num_rows = 100
current_date = datetime.date.today() - datetime.timedelta(days=365 * 2)

# Date 10 years ago
ten_years_ago = current_date - datetime.timedelta(days=365 * 10)

# Generate a fake date within the last decade
fake_date_last_decade = fake.date_between(
    start_date=ten_years_ago, end_date=current_date
)

# Define the first quarter of columns
first_quarter_columns = [
    "UNIQUE_ID",
    "Term",
    "College ID",
    "Age",
    "CAS CAA English SUM 1",
    "CAS CAA Foreign Language 1 SUM 1",
    "CAS CAA Foreign Language 2 SUM 1",
    "CAS CAA Math SUM 1",
    "CAS CAA Performing Arts SUM 1",
    "CAS CAA Science SUM 1",
    "CAS CAA Social Studies SUM 1",
    "CAS College Admissions Avg (CAA) SUM 1",
    "CAS HS  Units English SUM 1",
    "CAS HS Units Foreign Language 1 SUM 1",
    "CAS HS Units Foreign Language 2 SUM 1",
    "CAS HS Units Math SUM 1",
    "CAS HS Units Performing Arts SUM 1",
    "CAS HS Units Science SUM 1",
    "CAS HS Units Social Studies SUM 1",
    "CAS HS Units Total SUM 1",
    "CAS Regents American History",
    "CAS Regents Biology",
    "CAS Regents Chemistry",
    "CAS Regents Common Core Alg1",
    "CAS Regents Common Core Alg2",
    "CAS Regents Common Core Geometry",
    "CAS Regents Earth Science",
    "CAS Regents English",
    "CAS Regents English New",
    "CAS Regents Geometry",
    "CAS Regents Global History",
    "CAS Regents Integ Alg",
    "CAS Regents Marine Science",
    "CAS Regents Math A",
    "CAS Regents Math B",
    "CAS Regents Math Sequential 1",
    "CAS Regents Math Sequential 2",
    "CAS Regents Math Sequential 3",
    "CAS Regents Physics",
    "CAS Regents Trigonometry",
    "CAS SAT Math",
    "CAS SAT Total",
    "CAS SAT Verbal",
    "CAS SAT Writing",
    "CS Total Income Amt SUM",
    "College Program Title",
    "Credits Cumulative Earned Local Perf",
    "Credits Cumulative Earned Local Show",
    "Credits Cumulative Earned Other Perf",
    "Credits Cumulative Earned Other Show",
    "Credits Cumulative Earned Total Perf",
    "Credits Cumulative Earned Total Show",
    "Credits Cumulative Earned Transfer Perf",
    "Credits Cumulative Earned Transfer Show",
    "Credits Semester Included in GPA Perf SUM",
    "Credits+Hours Semester All Courses Show SUM",
    "Credits+Hours Semester Completed Perf SUM",
    "Credits+Hours Semester Passed Perf SUM",
]

# Define the second quarter of columns
second_quarter_columns = [
    "Cs Fed Efc Amt SUM",
    "Cs Fed Need Base Aid Amt SUM",
    "Cs Fed Unmet Need Amt SUM",
    "Degree Earned Level Desc",
    "Degree Pursued Level Desc",
    "Ethnicity Imputed Group 2 Desc",
    "Exit Math CUNY Test Final Score",
    "Exit Reading CUNY Test Final Score",
    "Exit Writing CUNY Test Final Score",
    "File Name",
    "Full Part Type Desc",
    "GPA Cumulative Perf",
    "GPA Cumulative Show",
    "GPA Semester Perf",
    "Gender Desc",
    "Headcount SUM",
    "Headcount SUM 1",
    "IR Total FA Awd Amt SUM",
    "Initial All SKAT Passed Desc",
    "Initial Math CUNY Test Final Score",
    "Initial Reading ACT Test Final Score",
    "Initial Writing CUNY Test Final Score",
    "Ir Fed Work Study Nb Awd Amt SUM",
    "Ir Student Loans Nb Awd Amt SUM",
    "Ir Total Grants Nb Awd Amt SUM",
    "Major 1 College Program Title",
    "Major 1 NYSED Award 1 Desc",
    "Major 1 NYSED Award 2 Desc",
    "Major 1 NYSED Program Title",
    "NYSED Award Desc",
    "NYSED Program Title",
    "New Student Desc",
    "PELL_AMT_AWD",
    "Pell Flag",
    "SEEK CD Desc",
    "Semester Enrolled Desc",
    "Semester Graduated Desc",
    "TAP Flag",
    "TAP_AMT_AWD",
    "Value Points Semester Perf SUM",
    "Year Enrolled",
    "Year Graduated",
]

# Create empty lists for the second quarter of columns
data_second_quarter = {col: [] for col in second_quarter_columns}

data = {col: [] for col in first_quarter_columns}


# Generate synthetic data for the first quarter of columns
for _ in range(num_rows):
    data["UNIQUE_ID"].append(fake.unique.random_number(digits=12))
    data["Term"].append(fake_date_last_decade)
    data["College ID"].append(fake.random_int(min=1000, max=9999))
    data["Age"].append(random.randint(18, 30))
    data["CAS CAA English SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS CAA Foreign Language 1 SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS CAA Foreign Language 2 SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS CAA Math SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS CAA Performing Arts SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS CAA Science SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS CAA Social Studies SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS College Admissions Avg (CAA) SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS HS  Units English SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Foreign Language 1 SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Foreign Language 2 SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Math SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Performing Arts SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Science SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Social Studies SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS HS Units Total SUM 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=4)])
    )
    data["CAS Regents American History"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Biology"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Chemistry"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Common Core Alg1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Common Core Alg2"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Common Core Geometry"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Earth Science"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents English"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents English New"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Geometry"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Global History"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Integ Alg"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Marine Science"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Math A"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Math B"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Math Sequential 1"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Math Sequential 2"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Math Sequential 3"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Physics"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS Regents Trigonometry"].append(
        random.choice([np.nan, fake.random_int(min=1, max=100)])
    )
    data["CAS SAT Math"].append(
        random.choice([np.nan, fake.random_int(min=200, max=800)])
    )
    data["CAS SAT Total"].append(
        random.choice([np.nan, fake.random_int(min=600, max=1600)])
    )
    data["CAS SAT Verbal"].append(
        random.choice([np.nan, fake.random_int(min=200, max=800)])
    )
    data["CAS SAT Writing"].append(
        random.choice([np.nan, fake.random_int(min=200, max=800)])
    )
    data["CS Total Income Amt SUM"].append(fake.random_int(min=10000, max=50000))
    data["College Program Title"].append(
        random.choice(df["College Program Title"].unique())
    )
    data["Credits Cumulative Earned Local Perf"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Local Show"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Other Perf"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Other Show"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Total Perf"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Total Show"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Transfer Perf"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Cumulative Earned Transfer Show"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits Semester Included in GPA Perf SUM"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data["Credits+Hours Semester All Courses Show SUM"].append(
        random.choice([np.nan, fake.random_int(min=15, max=100)])
    )
    data["Credits+Hours Semester Completed Perf SUM"].append(
        random.choice([np.nan, fake.random_int(min=25, max=100)])
    )
    data["Credits+Hours Semester Passed Perf SUM"].append(
        random.choice([np.nan, fake.random_int(min=20, max=100)])
    )
    data_second_quarter["Cs Fed Efc Amt SUM"].append(
        fake.random_int(min=1000, max=5000)
    )
    data_second_quarter["Cs Fed Need Base Aid Amt SUM"].append(
        fake.random_int(min=1000, max=5000)
    )
    data_second_quarter["Cs Fed Unmet Need Amt SUM"].append(
        fake.random_int(min=0, max=2000)
    )
    data_second_quarter["Degree Earned Level Desc"].append(np.nan)
    data_second_quarter["Degree Pursued Level Desc"].append(
        random.choice(["Associate", "Bachelor's"])
    )
    data_second_quarter["Ethnicity Imputed Group 2 Desc"].append(
        random.choice(
            ["White, Non-Hispanic", "Hispanic", "Asian", "Black, Non-Hispanic", "Other"]
        )
    )
    data_second_quarter["Exit Math CUNY Test Final Score"].append(
        fake.random_int(min=0, max=100)
    )
    data_second_quarter["Exit Reading CUNY Test Final Score"].append(
        fake.random_int(min=0, max=100)
    )
    data_second_quarter["Exit Writing CUNY Test Final Score"].append(
        fake.random_int(min=0, max=100)
    )
    data_second_quarter["File Name"].append(fake.file_name())
    data_second_quarter["Full Part Type Desc"].append(
        random.choice(["FULL-TIME", "PART-TIME"])
    )
    data_second_quarter["GPA Cumulative Perf"].append(
        round(random.uniform(2.0, 4.0), 2)
    )
    data_second_quarter["GPA Cumulative Show"].append(
        round(random.uniform(2.0, 4.0), 2)
    )
    data_second_quarter["GPA Semester Perf"].append(round(random.uniform(2.0, 4.0), 2))
    data_second_quarter["Gender Desc"].append(
        random.choice(["Male", "Female", "Other"])
    )
    data_second_quarter["Headcount SUM"].append(fake.random_int(min=1, max=5))
    data_second_quarter["Headcount SUM 1"].append(fake.random_int(min=1, max=5))
    data_second_quarter["IR Total FA Awd Amt SUM"].append(
        fake.random_int(min=1000, max=10000)
    )
    data_second_quarter["Initial All SKAT Passed Desc"].append(
        random.choice(["Exempt/Passed All", "Not Exempt/Passed All"])
    )
    data_second_quarter["Initial Math CUNY Test Final Score"].append(
        fake.random_int(min=0, max=100)
    )
    data_second_quarter["Initial Reading ACT Test Final Score"].append(
        fake.random_int(min=0, max=36)
    )
    data_second_quarter["Initial Writing CUNY Test Final Score"].append(
        fake.random_int(min=0, max=100)
    )
    data_second_quarter["Ir Fed Work Study Nb Awd Amt SUM"].append(
        fake.random_int(min=0, max=2000)
    )
    data_second_quarter["Ir Student Loans Nb Awd Amt SUM"].append(
        fake.random_int(min=0, max=20000)
    )
    data_second_quarter["Ir Total Grants Nb Awd Amt SUM"].append(
        fake.random_int(min=0, max=20000)
    )
    data_second_quarter["Major 1 College Program Title"].append(
        random.choice(df["Major 1 College Program Title"].unique())
    )
    data_second_quarter["Major 1 NYSED Award 1 Desc"].append(
        random.choice(df["Major 1 NYSED Award 1 Desc"].unique())
    )
    data_second_quarter["Major 1 NYSED Award 2 Desc"].append(
        random.choice(df["Major 1 NYSED Award 2 Desc"].unique())
    )
    data_second_quarter["Major 1 NYSED Program Title"].append(
        random.choice(df["Major 1 NYSED Program Title"].unique())
    )
    data_second_quarter["NYSED Award Desc"].append(
        random.choice(
            [np.nan, "BS", "BA", "BBA", "B TECH", "UNKNOWN", "BPS", "BFA", "BE", "BSED"]
        )
    )
    data_second_quarter["NYSED Program Title"].append(
        random.choice(
            [
                np.nan,
                "ECONOMICS",
                "CRIMINAL JUSTICE",
                "LAW, POLICE SCIENCE, AND CRIMINAL JUSTIC",
                "LAW AND SOCIETY",
                "FIRE SCIENCE",
                "PSYCHOLOGY",
                "POLICE STUDIES",
                "PHYSICAL SCIENCE PRE-BA LIBERAL ARTS",
                "CELLULAR AND MOLECULAR BIOLOGY",
                "HUMAN SERVICES: MENTAL HEALTH",
                "LIB ARTS: SOCIAL SCIENCES & HUMANITIES",
                "CRIMINOLOGY",
                "PROGRAMMING AND SYSTEMS",
                "LIBERAL ARTS",
                "CRIMINAL JUSTIC",
                "DEVIANT BEHAVIOR & SOCIAL CONTROL",
                "DIETETICS AND NUTRITION SCIENCE",
                "FORENSIC PSYCHOLOGY",
                "BUSINESS ADMINISTRATION",
                "FIRE SERVICE ADMINISTRATION",
                "HUMAN SERVICES",
                "PARALEGAL STUDIES",
                "LEGAL STUDIES",
                "PUBLIC ADMINISTRATION",
                "PUBLIC INTEREST PARALEGAL STUDIES",
                "ACCOUNTING",
                "CHILD CARE-EARLY CHILDHOOD EDUCATION",
                "SMALL BUSINESS/ENTREPRENEURSHIP",
                "LIBERAL ARTS: MATHEMATICS AND SCIENCE",
                "ENGLISH LITERATURE",
                "GOVERNMENT",
                "FINE ARTS",
                "COMPUTER INFORMATION SYSTEMS",
                "SECURITY MANAGEMENT",
                "PUBLIC AFFAIRS",
                "BUSINESS MANAGEMENT",
                "MARINE TECHNOLOGY",
                "MENTAL HEALTH AND HUMAN SERVICES",
                "FINANCE AND INVESTMENTS",
                "GOLBAL HISTORY",
                "LIBERAL ARTS AND SCIENCE",
                "NURSING",
                "INTERNATIONAL CRIMINAL JUSTICE",
                "ACCOUNTING FOR FORENSIC ACCOUNTING",
                "COMPUTER OPERATIONS",
                "FRAUD EXAMINATION AND FINANCIAL FORENSIC",
                "SPANISH",
                "GENDER STUDIES",
                "WRITING AND LITERATURE",
                "LIBERAL ARTS AND SCIENCES",
                "FORENSIC SCIENCE",
                "EMERGENCY MEDICAL TECHNICIAN/PARAMEDIC",
                "COMPUTER INFO SYS-CRIMINAL JUST&PUB ADMN",
                "COMPUTER DATA PROCESSING",
                "HUMAN SERVICES: CHILD DEVELOPMENT",
                "COMPUTER SCIENCE",
                "JUDICIAL STUDIES",
                "BUSINESS",
                "BIOLOGY",
                "CHEMISTRY",
                "ACCOUNTING FOR FORENSICS",
                "SCIENCE FOR FORENSICS",
                "SOCIOLOGY",
                "VETERINARY TECHNOLOGY",
                "PHILOSOPHY",
                "SECRETARIAL SCI/OFFICE ADMINISTRATION",
                "CUNY BACCALAUREATE",
                "CORPORATE AND CABLE COMMUNICATIONS",
                "CRIMINAL JUSTICE ADMINISTRATION AND PLAN",
                "SPANISH TEACHER",
                "JUSTICE STUDIES",
                "POLITICAL SCIENCE",
                "COMPUTER TECHNOLOGY",
                "EDUCATION ASSOCIATE",
                "COMMERCIAL PHOTOGRAPHY",
                "INFORMATION SYSTEMS MANAGEMENT",
                "UNCLASSIFIED",
                "HEALTH INFORMATION TECHNOLOGY",
                "HUMAN SERVICES: GERONTOLOGY",
                "BUSINESS COMMUNICATIONS",
                "SCIENCE",
                "OFFICE OPERATIONS",
                "NEW MEDIA TECHNOLOGY",
                "LATIN AMERICAN AND LATINA/O STUDIES",
                "URBAN STUDIES",
                "RADIOLOGIC TECHNOLOGY",
                "TRAVEL AND TOURISM",
                "LIB ARTS & SCIENCES - MATH & SCIENCE",
                "MARKETING, MANAGEMENT, SALES",
                "FINE AND PERFORMING ARTS",
                "FASHION MERCHANDISING",
                "CORRECTIONAL STUDIES",
                "ANTHROPOLOGY",
                "THEATRE ARTS",
                "BUSINESS, MANAGEMENT AND FINANCE",
                "POLITICAL SCIENCE AND GOVERNMENT",
                "MUSIC ELECTRONIC TECHNOLOGY",
                "GEOGRAPHIC INFORMATION SCIENCE",
                "BROADCASTING TECHNOLOGY & MANAGEMENT",
                "THEATRE",
                "FIRE AND EMERGENCY SERVICE",
                "COMPUTER SYSTEMS",
                "MARKETING",
                "OFFICE ADMINISTRATION AND TECHNOLOGY",
                "COMPUTER PROGRAMMING",
                "INTERDISCIP LIB ARTS & SCI (WORKER ED)",
                "JOURNALISM AND PRINT MEDIA",
                "INTERDEPART CONCENTRATION IN ANTHROPOLOG",
                "GEOGRAPHY",
                'ROMANCE LANG "FRENCH,ITALIAN,SPANISH"',
                "SECRETARIAL SCIENCE--MEDICAL",
                "INTERNAL ACCOUNTING",
                "COMPUTING AND MANAGEMENT",
                "MANAGEMENT",
                "DUAL DEGREE-PUBLIC ADMINISTRATION",
                "MULTIMEDIA PROGRAMMING AND DESIGN",
                "POLITICAL SCIENCE AND GOV SOCIAL STUDIES",
                "ADVERTISING ART & COMPUTER GRAPHICS",
                "EDUCATION ASSOCIATE:THE BILINGUAL CHILD",
                "GRAPHIC DESIGN AND ILLUSTRATION",
                "SOCIAL WORK",
                "CHILDHOOD EDU (1-6)",
                "HEALTH SERVICES ADMINISTRATION",
                "ENGLISH",
                "GERONTOLOGY",
                "AUTOMOTIVE TECHNOLOGY",
                "HEALTH SCIENCES",
                "COMMUNITY SCHOOL/HEALTH EDUCATION",
                "RADIOLOGIC TECHNOLOGY & MEDICAL IMAGING",
                "PUBLIC ACCOUNTING & BUSNS MGMNT & FINANC",
                "HISTORY",
                "DIGITAL ART AND DESIGN",
                "MICROCOMPUTER SYSTEMS & APPLICATIONS",
                "JOURNALISM",
                "EARLY CHILDHOOD EDUCATION/CHILD CARE",
                "COMMUNICATION AND CULTURE",
                "DESIGN DRAFTING AND COMPUTER GRAPHICS",
                "SPEECH PATHOLOGY & AUDIOLOGY",
                "MEDICAL TECHNOLOGY",
                "LEGAL ASSISTANT STUDIES",
                "COMMUNICATION ARTS",
                "ENGINEERING SCIENCE",
                "CHEMICAL TECHNOLOGY",
                "ENGLISH TEACHER",
                "TELEVISION TECHNOLOGY",
                "GAME DESIGN",
                "COMPUTER METHODOLOGY",
                "NURSING (ONLINE DEGREE)",
                "SURGICAL TECHNOLOGY",
                "NUTRITION AND EXERCISE SCIENCES",
                "COMMUNITY HEALTH",
                "CULINARY ARTS",
                "MEDIA ARTS",
                "SPEECH COMMUNICATION",
                "PHYSICAL THERAPIST ASSISTANT",
                "FAMILY AND CONSUMER SCI TEACHER K-12",
                "MATHEMATICS",
                "GEOLOGY",
                "FILM AND TV STUDIES",
                "EMERGING MEDIA TECHNOLOGIES",
                "DIETETICS, FOODS & NUTRITION",
                "SECRETARIAL STUDIES",
                "MEDIA STUDIES",
                "FINANCE",
                "STUDIO ART (24 CREDIT MAJOR)",
                "MARKETING MANAGEMENT",
                "AREA STUDIES-BLACK-PUERTO RICAN-JEWISH",
                "EARLY CHILDHOOD/EARLY CHILD SPEC ED",
                "ACCOUNTANCY",
                "SOCIAL STUDIES TEACHER",
                "SECRETARIAL SCIENCE-EXECUTIVE",
                "GERONTOLOGICAL STUDIES AND SERVICES",
                "EARLY CHILDHOOD EDUCATION",
                "INFORMATION SYSTEMS (ONLINE DEGREE)",
                "THERAPEUTIC RECREATION",
                "OPHTHALMIC DISPENSING",
                "CHEMICAL DEPENDENCY COUNSELING",
                "COMMUNICATIONS",
                "POLICE SCIENCE",
                "BLACK AND PUERTO RICAN STUDIES",
                "EXERCISE SCIENCE",
                "ART",
                "FILM",
                "WEBSITE DEVELOPMENT AND ADM.",
                "SECRETARIAL SCI-ADMINISTRATIVE SECRETARY",
                "DIGITAL DESIGN AND ANIMATION",
                "COMMUNICATION STUDIES",
                "ELECTRONIC ENGINEERING TECHNOLOGY",
                "COMPUTER ENGINEERING TECHNOLOGY",
                "COMMUNICATION DESIGN",
                "MEDICAL OFFICE ASSISTANT",
                "LIBARAL ARTS",
                "COMMUNITY HEALTH EDUCATION",
                "MATHEMATICS, GRADES 7-12",
                "INFORMATION TECHNOLOGY",
                "DENTAL HYGIENE",
                "OFFICE AUTOMATION",
                "CHEMISTRY MAJOR II",
                "EDUCATION STUDIES",
                "TOXICOLOGY",
                "MASSAGE THERAPY",
                "ENVIRONMENTAL CONTROL TECHNOLOGY",
                "CORRECTION ADMINISTRATION",
                "ACTUARIAL SCIENCE",
                "MORTUARY SCIENCE",
                "TEACHER EDUCATION",
                "APPIED MANAGEMENT",
                "MICROCOMPUTERS FOR BUSINESS",
                "TELEVISION AND RADIO",
                "STATISTICS",
                "LIBERAL ARTS & SCIENCES",
                "LABOR STUDIES",
                "PHARMACEUTICAL MANUFACTURING TECH",
                "DESIGN",
                "MUSIC RECORDING TECHNOLOGY",
                "STAGE TECHNOLOGY",
                "GRAPHIC DESIGN",
                "PHYSICAL EDUCATION",
                "SPORTS, FITNESS & THERAPEUTIC RECREATION",
                "MEDIA COMMUNICATIONS STUDIES",
                "ANIMATION AND MOTION GRAPHICS",
                "SOCIOLOGY / ANTHROPOLOGY",
                "BIOLOGICAL SCIENCE - MAJOR I",
                "EDUCATION (PLACE HOLDER)",
                "PROFESSIONAL AND TECHNICAL WRITING",
                "MEDICAL LABORATORY SCIENCES",
                "SPANISH (BILING)",
                "MEDIA AND COMMUNICATION SCI & DISORDERS",
                "MEDICAL LABORATORY TECHNICIAN",
                "SPECIAL EDUCATION AND CHILDHOOD",
                "DENTAL LABORATORY TECHNOLOGY",
                "ARCHITECHTURAL TECHNOLOGY",
                "LATIN AMERICAN AND CARIBBEAN S",
                "COMPARATIVE LITERATURE",
                "STUDIO ART (42 CREDIT MAJOR)",
                "HOME ECONOMICS",
                "CHILDHOOD EDU (GRADES 1-6)",
                "TELECOMMUNICATIONS TECHNOLOGY: NYNEX",
                "DANCE",
                "HUMAN RELATIONS",
                "PHYSICAL EDU AND EXERCISE SCIENCE",
                "OCCUPATIONAL THERAPY ASSISTANT",
                "MANAGEMENT AND ADMINISTRATION",
                "MODERN LANGUAGES",
                "COMMUNICATION",
                "ENVIRONMENTAL SCIENCE",
                "BUSINESS ADMINISTRATION: FINANCE",
                "EARTH SYSTEMS SCI AND ENVIRMNT ENG",
                "SCIENCE, LETTERS, SOCIETY",
                "CHILDHOOD EDU.",
                "ENGLISH LANGUAGE ARTS",
                "DIGITAL MUSIC",
                "PHYSICAL EDUCATION, RECREATION, AND RECR",
                "MEDICAL LAB SCI - CLINICAL SCI",
                "ELECTRICAL TECHNOLOGY",
                "RELIGION",
                "CREATIVE WRITING",
                "DISABILITY STUDIES",
                "STUDIO ART",
                "MUSIC",
                "HEALTH SCIENCE",
                "EMERGENCY MEDICAL SERVICES",
                "HOSPITALITY MANAGEMENT",
                "PUBLIC COMMUNITY HEALTH",
                "AVIATION MGMT",
                "CHILDHOOD EDUCATION",
                "CINEMA STUDIES",
                "HEALTH & NUTRITION SCIENCES",
                "BUSINESS ADMIN: INTERNATIONAL BUS",
                "ART HISTORY",
                "DANCE IN THE PHYSICAL EDUCATION PROG",
                "HEALTH INFORMATION MANAGMENT",
                "FINANCIAL MANAGEMENT",
                "NURSING (RN)",
            ]
        )
    )
    data_second_quarter["New Student Desc"].append(random.choice(["Yes", "No"]))
    data_second_quarter["PELL_AMT_AWD"].append(fake.random_int(min=1000, max=5000))
    data_second_quarter["Pell Flag"].append(random.choice(["Yes", "No"]))
    data_second_quarter["SEEK CD Desc"].append(
        random.choice(["Regular", np.nan, "SEEK", "CD Prong I", "CD Bilingual"])
    )
    data_second_quarter["Semester Enrolled Desc"].append(
        random.choice(["Full-Time", "Part-Time"])
    )
    data_second_quarter["Semester Graduated Desc"].append(
        random.choice(["Graduated", "Not Graduated"])
    )
    data_second_quarter["TAP Flag"].append(random.choice(["Yes", "No"]))
    data_second_quarter["TAP_AMT_AWD"].append(fake.random_int(min=1000, max=5000))
    data_second_quarter["Value Points Semester Perf SUM"].append(
        round(random.uniform(0.0, 4.0), 2)
    )
    data_second_quarter["Year Enrolled"].append(fake.random_int(min=2010, max=2020))
    data_second_quarter["Year Graduated"].append(np.nan)

# Create a DataFrame for the second quarter of columns
df_first_quarter = pd.DataFrame(data)
df_second_quarter = pd.DataFrame(data_second_quarter)
joined = pd.concat([df_first_quarter, df_second_quarter], axis=1)
# Print the first few rows of the DataFrame for the second quarter

# COMMAND ----------

joined.head()

# COMMAND ----------

len(joined["UNIQUE_ID"].unique())

# COMMAND ----------

joined.to_csv("synthetic_dataset.csv", index=False)

# COMMAND ----------

df["Initial All SKAT Passed Desc"].unique()

# COMMAND ----------

len(df_columns)

# COMMAND ----------

len(joined.columns)

# COMMAND ----------

# Assuming df_columns and joined.columns are lists of column names
set(df_columns) - set(joined.columns)


# COMMAND ----------

joined.columns

# COMMAND ----------

df_columns

# COMMAND ----------

df["Term"].unique()

# COMMAND ----------

joined["SEEK CD Desc"].unique()

# COMMAND ----------

joined["Term"].unique()

# COMMAND ----------


def generate_random_string(n):
    """Generate random alphanumeric string of length n"""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def generate_random_date(start, end):
    """Generate a random date between start and end"""
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_random_decimal(min_value, max_value, n):
    """Generate a random value between min_value and max_value with n decimal places"""
    return round(random.uniform(min_value, max_value), n)


# List of possible College IDs and Program Titles
college_ids = [2, 3, 4, 8, 9, 12, 13, 11]
program_titles = [
    "POLICE SCIENCE",
    "BUSINESS ADMINISTRATION",
    "SECURITY MANAGEMENT",
    "PUBLIC ADMINISTRATION",
    "LAW AND SOCIETY",
    "WRITING AND LITERATURE",
    "LIBERAL ARTS AND SCIENCE",
    "RADIOLOGIC TECHNOLOGY",
    "LIB ARTS: SOCIAL SCIENCES & HUMANITIES",
    "FORENSIC PSYCHOLOGY",
    "COMPUTER OPERATIONS",
]

# Generate unique IDs
unique_ids = [generate_random_string(8) for _ in range(20)]

# Open the CSV file for writing
with open("0424_23_synthetic_data_2.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "UNIQUE_ID",
            "Term",
            "College ID",
            "Year Graduated",
            "Semester Graduated Desc",
            "Full Part Type Desc",
            "Credits+Hours Semester All Courses Show SUM",
            "Credits+Hours Semester Completed Perf SUM",
            "Credits+Hours Semester Passed Perf SUM",
            "Credits Semester Included in GPA Perf SUM",
            "IR Total FA Awd Amt SUM",
            "Ir Total Grants Nb Awd Amt SUM",
            "Ir Student Loans Nb Awd Amt SUM",
            "Ir Fed Work Study Nb Awd Amt SUM",
            "PELL_AMT_AWD",
            "Pell Flag",
            "TAP Flag",
            "TAP_AMT_AWD",
            "GPA Semester Perf",
            "GPA Cumulative Perf",
            "GPA Cumulative Show",
            "Major 1 NYSED Program Title",
            "Degree Earned Level Desc",
            "Value Points Semester Perf SUM",
        ]
    )

    # Generate multiple rows of data for each UNIQUE_ID and write each to the CSV file
    for unique_id in unique_ids:
        num_records = random.randint(2, 5)  # Each student has 2 to 5 records
        terms = sorted(
            [
                generate_random_date(datetime(2010, 1, 1), datetime(2020, 6, 30))
                for _ in range(num_records)
            ]
        )
        sorted_terms = sorted(terms)

        for term in sorted_terms:
            term_str = term.strftime("%d-%m-%Y")
            college_id = random.choice(college_ids)
            year_graduated = ""
            semester_graduated_desc = ""
            full_part_type_desc = random.choice(["PART-TIME", "FULL-TIME"])
            credits_hours_semester_all_courses_show_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_completed_perf_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_passed_perf_sum = generate_random_decimal(3, 12, 1)
            credits_semester_included_in_gpa_perf_sum = generate_random_decimal(
                6, 12, 1
            )
            ir_total_fa_awd_amt_sum = generate_random_decimal(2800, 8200, 2)
            ir_total_grants_nb_awd_amt_sum = generate_random_decimal(950, 7777, 2)
            ir_student_loans_nb_awd_amt_sum = generate_random_decimal(3850, 5500, 2)
            ir_fed_work_study_nb_awd_amt_sum = 0
            pell_amt_awd = generate_random_decimal(650, 2600, 2)
            pell_flag = random.choice(["Yes", "No"])
            tap_flag = random.choice(["Yes", "No"])
            tap_amt_awd = generate_random_decimal(0, 1300, 2)
            gpa_semester_perf = generate_random_decimal(2.4, 3.1, 2)
            gpa_cumulative_perf = generate_random_decimal(2.4, 3, 2)
            gpa_cumulative_show = generate_random_decimal(2.4, 3.1, 2)
            major_1_nysed_program_title = random.choice(program_titles)
            degree_earned_level_desc = ""
            value_points_semester_perf_sum = generate_random_decimal(9, 38, 1)

            writer.writerow(
                [
                    unique_id,
                    term_str,
                    college_id,
                    year_graduated,
                    semester_graduated_desc,
                    full_part_type_desc,
                    credits_hours_semester_all_courses_show_sum,
                    credits_hours_semester_completed_perf_sum,
                    credits_hours_semester_passed_perf_sum,
                    credits_semester_included_in_gpa_perf_sum,
                    ir_total_fa_awd_amt_sum,
                    ir_total_grants_nb_awd_amt_sum,
                    ir_student_loans_nb_awd_amt_sum,
                    ir_fed_work_study_nb_awd_amt_sum,
                    pell_amt_awd,
                    pell_flag,
                    tap_flag,
                    tap_amt_awd,
                    gpa_semester_perf,
                    gpa_cumulative_perf,
                    gpa_cumulative_show,
                    major_1_nysed_program_title,
                    degree_earned_level_desc,
                    value_points_semester_perf_sum,
                ]
            )


# COMMAND ----------

df = pd.read_csv("0424_23_synthetic_data_2.csv")
risk_df = df.groupby("UNIQUE_ID")["Term"].agg([min, max]).reset_index()

# COMMAND ----------

df

# COMMAND ----------

risk_df

# COMMAND ----------


def generate_sequential_years(start_year, end_year, num):
    """Generate a sequence of random years in ascending order"""
    current_year = start_year
    years = []
    for _ in range(num):
        next_year = current_year + random.randint(1, 2)  # Increment year by 1 or 2
        if next_year > end_year:
            break
        years.append(next_year)
        current_year = next_year
    return years


# List of possible College IDs and Program Titles
college_ids = [2, 3, 4, 8, 9, 12, 13, 11]
program_titles = [
    "POLICE SCIENCE",
    "BUSINESS ADMINISTRATION",
    "SECURITY MANAGEMENT",
    "PUBLIC ADMINISTRATION",
    "LAW AND SOCIETY",
    "WRITING AND LITERATURE",
    "LIBERAL ARTS AND SCIENCE",
    "RADIOLOGIC TECHNOLOGY",
    "LIB ARTS: SOCIAL SCIENCES & HUMANITIES",
    "FORENSIC PSYCHOLOGY",
    "COMPUTER OPERATIONS",
]

# Generate unique IDs
unique_ids = [generate_random_string(8) for _ in range(20)]

# Open the CSV file for writing
with open("0424_23_synthetic_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "UNIQUE_ID",
            "Term",
            "College ID",
            "Year Graduated",
            "Semester Graduated Desc",
            "Full Part Type Desc",
            "Credits+Hours Semester All Courses Show SUM",
            "Credits+Hours Semester Completed Perf SUM",
            "Credits+Hours Semester Passed Perf SUM",
            "Credits Semester Included in GPA Perf SUM",
            "IR Total FA Awd Amt SUM",
            "Ir Total Grants Nb Awd Amt SUM",
            "Ir Student Loans Nb Awd Amt SUM",
            "Ir Fed Work Study Nb Awd Amt SUM",
            "PELL_AMT_AWD",
            "Pell Flag",
            "TAP Flag",
            "TAP_AMT_AWD",
            "GPA Semester Perf",
            "GPA Cumulative Perf",
            "GPA Cumulative Show",
            "Major 1 NYSED Program Title",
            "Degree Earned Level Desc",
            "Value Points Semester Perf SUM",
        ]
    )

    # Generate multiple rows of data for each UNIQUE_ID and write each to the CSV file
    for unique_id in unique_ids:
        num_records = random.randint(2, 5)  # Each student has 2 to 5 records
        terms = generate_sequential_years(2010, 2020, num_records)

        for term in terms:
            college_id = random.choice(college_ids)
            year_graduated = ""
            semester_graduated_desc = ""
            full_part_type_desc = random.choice(["PART-TIME", "FULL-TIME"])
            credits_hours_semester_all_courses_show_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_completed_perf_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_passed_perf_sum = generate_random_decimal(3, 12, 1)
            credits_semester_included_in_gpa_perf_sum = generate_random_decimal(
                6, 12, 1
            )
            ir_total_fa_awd_amt_sum = generate_random_decimal(2800, 8200, 2)
            ir_total_grants_nb_awd_amt_sum = generate_random_decimal(950, 7777, 2)
            ir_student_loans_nb_awd_amt_sum = generate_random_decimal(3850, 5500, 2)
            ir_fed_work_study_nb_awd_amt_sum = 0
            pell_amt_awd = generate_random_decimal(650, 2600, 2)
            pell_flag = random.choice(["Yes", "No"])
            tap_flag = random.choice(["Yes", "No"])
            tap_amt_awd = generate_random_decimal(0, 1300, 2)
            gpa_semester_perf = generate_random_decimal(2.4, 3.1, 2)
            gpa_cumulative_perf = generate_random_decimal(2.4, 3, 2)
            gpa_cumulative_show = generate_random_decimal(2.4, 3.1, 2)
            major_1_nysed_program_title = random.choice(program_titles)
            degree_earned_level_desc = ""
            value_points_semester_perf_sum = generate_random_decimal(9, 38, 1)

            writer.writerow(
                [
                    unique_id,
                    term,
                    college_id,
                    year_graduated,
                    semester_graduated_desc,
                    full_part_type_desc,
                    credits_hours_semester_all_courses_show_sum,
                    credits_hours_semester_completed_perf_sum,
                    credits_hours_semester_passed_perf_sum,
                    credits_semester_included_in_gpa_perf_sum,
                    ir_total_fa_awd_amt_sum,
                    ir_total_grants_nb_awd_amt_sum,
                    ir_student_loans_nb_awd_amt_sum,
                    ir_fed_work_study_nb_awd_amt_sum,
                    pell_amt_awd,
                    pell_flag,
                    tap_flag,
                    tap_amt_awd,
                    gpa_semester_perf,
                    gpa_cumulative_perf,
                    gpa_cumulative_show,
                    major_1_nysed_program_title,
                    degree_earned_level_desc,
                    value_points_semester_perf_sum,
                ]
            )


# COMMAND ----------

df = pd.read_csv("0424_23_synthetic_data.csv")
risk_df = df.groupby("UNIQUE_ID")["Term"].agg([min, max]).reset_index()

# COMMAND ----------

risk_df

# COMMAND ----------


def generate_sequential_dates(start, end, num):
    """Generate a sequence of random dates in ascending order"""
    current_date = start
    dates = []
    for _ in range(num):
        next_date = current_date + timedelta(
            days=random.randint(30, 365)
        )  # Add 30 to 365 days to the current date
        if next_date > end:
            break
        dates.append(next_date)
        current_date = next_date
    return dates


# List of possible College IDs and Program Titles
college_ids = [2, 3, 4, 8, 9, 12, 13, 11]
program_titles = [
    "POLICE SCIENCE",
    "BUSINESS ADMINISTRATION",
    "SECURITY MANAGEMENT",
    "PUBLIC ADMINISTRATION",
    "LAW AND SOCIETY",
    "WRITING AND LITERATURE",
    "LIBERAL ARTS AND SCIENCE",
    "RADIOLOGIC TECHNOLOGY",
    "LIB ARTS: SOCIAL SCIENCES & HUMANITIES",
    "FORENSIC PSYCHOLOGY",
    "COMPUTER OPERATIONS",
]

# Generate unique IDs
unique_ids = [generate_random_string(8) for _ in range(20)]

# Open the CSV file for writing
with open("0424_23_synthetic_data_1.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "UNIQUE_ID",
            "Term",
            "College ID",
            "Year Graduated",
            "Semester Graduated Desc",
            "Full Part Type Desc",
            "Credits+Hours Semester All Courses Show SUM",
            "Credits+Hours Semester Completed Perf SUM",
            "Credits+Hours Semester Passed Perf SUM",
            "Credits Semester Included in GPA Perf SUM",
            "IR Total FA Awd Amt SUM",
            "Ir Total Grants Nb Awd Amt SUM",
            "Ir Student Loans Nb Awd Amt SUM",
            "Ir Fed Work Study Nb Awd Amt SUM",
            "PELL_AMT_AWD",
            "Pell Flag",
            "TAP Flag",
            "TAP_AMT_AWD",
            "GPA Semester Perf",
            "GPA Cumulative Perf",
            "GPA Cumulative Show",
            "Major 1 NYSED Program Title",
            "Degree Earned Level Desc",
            "Value Points Semester Perf SUM",
        ]
    )

    # Generate multiple rows of data for each UNIQUE_ID and write each to the CSV file
    for unique_id in unique_ids:
        num_records = random.randint(2, 5)  # Each student has 2 to 5 records
        start_date = datetime(2010, 1, 1)
        terms = generate_sequential_dates(
            start_date, datetime(2020, 6, 30), num_records
        )

        for term_date in terms:
            term_str = term_date.strftime("%d-%m-%Y")
            college_id = random.choice(college_ids)
            year_graduated = ""
            semester_graduated_desc = ""
            full_part_type_desc = random.choice(["PART-TIME", "FULL-TIME"])
            credits_hours_semester_all_courses_show_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_completed_perf_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_passed_perf_sum = generate_random_decimal(3, 12, 1)
            credits_semester_included_in_gpa_perf_sum = generate_random_decimal(
                6, 12, 1
            )
            ir_total_fa_awd_amt_sum = generate_random_decimal(2800, 8200, 2)
            ir_total_grants_nb_awd_amt_sum = generate_random_decimal(950, 7777, 2)
            ir_student_loans_nb_awd_amt_sum = generate_random_decimal(3850, 5500, 2)
            ir_fed_work_study_nb_awd_amt_sum = 0
            pell_amt_awd = generate_random_decimal(650, 2600, 2)
            pell_flag = random.choice(["Yes", "No"])
            tap_flag = random.choice(["Yes", "No"])
            tap_amt_awd = generate_random_decimal(0, 1300, 2)
            gpa_semester_perf = generate_random_decimal(2.4, 3.1, 2)
            gpa_cumulative_perf = generate_random_decimal(2.4, 3, 2)
            gpa_cumulative_show = generate_random_decimal(2.4, 3.1, 2)
            major_1_nysed_program_title = random.choice(program_titles)
            degree_earned_level_desc = ""
            value_points_semester_perf_sum = generate_random_decimal(9, 38, 1)

            writer.writerow(
                [
                    unique_id,
                    term_str,
                    college_id,
                    year_graduated,
                    semester_graduated_desc,
                    full_part_type_desc,
                    credits_hours_semester_all_courses_show_sum,
                    credits_hours_semester_completed_perf_sum,
                    credits_hours_semester_passed_perf_sum,
                    credits_semester_included_in_gpa_perf_sum,
                    ir_total_fa_awd_amt_sum,
                    ir_total_grants_nb_awd_amt_sum,
                    ir_student_loans_nb_awd_amt_sum,
                    ir_fed_work_study_nb_awd_amt_sum,
                    pell_amt_awd,
                    pell_flag,
                    tap_flag,
                    tap_amt_awd,
                    gpa_semester_perf,
                    gpa_cumulative_perf,
                    gpa_cumulative_show,
                    major_1_nysed_program_title,
                    degree_earned_level_desc,
                    value_points_semester_perf_sum,
                ]
            )


# COMMAND ----------

df = pd.read_csv("0424_23_synthetic_data_1.csv")
risk_df = df.groupby("UNIQUE_ID")["Term"].agg([min, max]).reset_index()

# COMMAND ----------

risk_df

# COMMAND ----------


def generate_sequential_dates(start_year, end_year, latest_possible_date, num_records):
    """Generate a sequence of random dates in ascending order, starting within a given range"""
    start_date = generate_random_date(
        datetime(start_year, 1, 1), datetime(end_year, 12, 31)
    )
    dates = [start_date]
    for _ in range(1, num_records):
        next_date = dates[-1] + timedelta(
            days=random.randint(30, 365)
        )  # Add 30 to 365 days to the last date
        if next_date > latest_possible_date:
            break
        dates.append(next_date)
    return dates


# List of possible College IDs and Program Titles
college_ids = [2, 3, 4, 8, 9, 12, 13, 11]
program_titles = [
    "POLICE SCIENCE",
    "BUSINESS ADMINISTRATION",
    "SECURITY MANAGEMENT",
    "PUBLIC ADMINISTRATION",
    "LAW AND SOCIETY",
    "WRITING AND LITERATURE",
    "LIBERAL ARTS AND SCIENCE",
    "RADIOLOGIC TECHNOLOGY",
    "LIB ARTS: SOCIAL SCIENCES & HUMANITIES",
    "FORENSIC PSYCHOLOGY",
    "COMPUTER OPERATIONS",
]

# Generate unique IDs
unique_ids = [
    generate_random_string(8) for _ in range(67)
]  # Approximately 67 unique IDs

# Open the CSV file for writing
with open("0424_23_synthetic_data_5.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "UNIQUE_ID",
            "Term",
            "College ID",
            "Year Graduated",
            "Semester Graduated Desc",
            "Full Part Type Desc",
            "Credits+Hours Semester All Courses Show SUM",
            "Credits+Hours Semester Completed Perf SUM",
            "Credits+Hours Semester Passed Perf SUM",
            "Credits Semester Included in GPA Perf SUM",
            "IR Total FA Awd Amt SUM",
            "Ir Total Grants Nb Awd Amt SUM",
            "Ir Student Loans Nb Awd Amt SUM",
            "Ir Fed Work Study Nb Awd Amt SUM",
            "PELL_AMT_AWD",
            "Pell Flag",
            "TAP Flag",
            "TAP_AMT_AWD",
            "GPA Semester Perf",
            "GPA Cumulative Perf",
            "GPA Cumulative Show",
            "Major 1 NYSED Program Title",
            "Degree Earned Level Desc",
            "Value Points Semester Perf SUM",
        ]
    )

    for unique_id in unique_ids:
        num_records = random.randint(10, 15)  # Between 10 and 15 records per unique ID
        terms = generate_sequential_dates(
            2010, 2018, datetime(2020, 12, 31), num_records
        )

        for term in terms:
            term_str = term.strftime("%Y-%m-%d")
            college_id = random.choice(college_ids)
            year_graduated = ""
            semester_graduated_desc = ""
            full_part_type_desc = random.choice(["PART-TIME", "FULL-TIME"])
            credits_hours_semester_all_courses_show_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_completed_perf_sum = generate_random_decimal(
                6, 13, 1
            )
            credits_hours_semester_passed_perf_sum = generate_random_decimal(3, 12, 1)
            credits_semester_included_in_gpa_perf_sum = generate_random_decimal(
                6, 12, 1
            )
            ir_total_fa_awd_amt_sum = generate_random_decimal(2800, 8200, 2)
            ir_total_grants_nb_awd_amt_sum = generate_random_decimal(950, 7777, 2)
            ir_student_loans_nb_awd_amt_sum = generate_random_decimal(3850, 5500, 2)
            ir_fed_work_study_nb_awd_amt_sum = 0
            pell_amt_awd = generate_random_decimal(650, 2600, 2)
            pell_flag = random.choice(["Yes", "No"])
            tap_flag = random.choice(["Yes", "No"])
            tap_amt_awd = generate_random_decimal(0, 1300, 2)
            gpa_semester_perf = generate_random_decimal(2.4, 3.1, 2)
            gpa_cumulative_perf = generate_random_decimal(2.4, 3, 2)
            gpa_cumulative_show = generate_random_decimal(2.4, 3.1, 2)
            major_1_nysed_program_title = random.choice(program_titles)
            degree_earned_level_desc = ""
            value_points_semester_perf_sum = generate_random_decimal(9, 38, 1)

            writer.writerow(
                [
                    unique_id,
                    term_str,
                    college_id,
                    year_graduated,
                    semester_graduated_desc,
                    full_part_type_desc,
                    credits_hours_semester_all_courses_show_sum,
                    credits_hours_semester_completed_perf_sum,
                    credits_hours_semester_passed_perf_sum,
                    credits_semester_included_in_gpa_perf_sum,
                    ir_total_fa_awd_amt_sum,
                    ir_total_grants_nb_awd_amt_sum,
                    ir_student_loans_nb_awd_amt_sum,
                    ir_fed_work_study_nb_awd_amt_sum,
                    pell_amt_awd,
                    pell_flag,
                    tap_flag,
                    tap_amt_awd,
                    gpa_semester_perf,
                    gpa_cumulative_perf,
                    gpa_cumulative_show,
                    major_1_nysed_program_title,
                    degree_earned_level_desc,
                    value_points_semester_perf_sum,
                ]
            )


# COMMAND ----------

df = pd.read_csv("0424_23_synthetic_data_5.csv")
risk_df = df.groupby("UNIQUE_ID")["Term"].agg([min, max]).reset_index()

# COMMAND ----------

risk_df

# COMMAND ----------

df
