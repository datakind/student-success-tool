DEFAULT_MIN_PASSING_GRADE = "1"
DEFAULT_COURSE_LEVEL_PATTERN = r"^(?P<course_level>\d)\d{2}(?:[A-Z]{,2})?$"
DEFAULT_PEAK_COVID_TERMS = {
    ("2019-20", "SPRING"), # Spring 2020
    ("2019-20", "SUMMER"), # Summer 2020
    ("2020-21", "FALL"), # Fall 2020
    ("2020-21", "WINTER"), # Winter 2020/2021
    ("2020-21", "SPRING"), # Spring 2021
    ("2020-21", "SUMMER"), # Summer 2021
}

NUM_COURSE_FEATURE_COL_PREFIX = "num_courses"
FRAC_COURSE_FEATURE_COL_PREFIX = "frac_courses"
