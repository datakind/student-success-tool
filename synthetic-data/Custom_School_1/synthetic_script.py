import csv
import random
import string
from datetime import datetime, timedelta


# TODO: duplicated in synthetic_dataset.py
def generate_random_string(n):
    """Generate random alphanumeric string of length n"""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


# TODO: duplicated in synthetic_dataset.py
def generate_random_date(start, end):
    """Generate a random date between start and end"""
    total_days = (end - start).days
    random_days = random.randint(0, total_days)
    return start + timedelta(days=random_days)


# TODO: duplicated in synthetic_dataset.py
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


# TODO: duplicated in synthetic_data.py
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
