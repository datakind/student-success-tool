import datetime
import functools as ft
import logging
import re
import typing as t

import pandas as pd

from .. import utils
from . import features
from .features.pdp import constants

LOGGER = logging.getLogger(__name__)


def make_student_term_dataset(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    min_passing_grade: float = constants.DEFAULT_MIN_PASSING_GRADE,
    min_num_credits_full_time: float = constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
    course_level_pattern: str = constants.DEFAULT_COURSE_LEVEL_PATTERN,
    core_terms: set[str] = constants.DEFAULT_CORE_TERMS,
    peak_covid_terms: set[tuple[str, str]] = constants.DEFAULT_PEAK_COVID_TERMS,
    key_course_subject_areas: t.Optional[list[str]] = None,
    key_course_ids: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Make a student-term dataset from raw cohort- and course-level datasets,
    including many features generated at the student-, course-, term-, section-,
    and student-term levels, as well as cumulatively over student-terms.

    Args:
        df_cohort: As output by :func:`dataio.read_raw_pdp_cohort_data_from_file()` .
        df_course: As output by :func:`dataio.read_raw_pdp_course_data_from_file()` .
        min_passing_grade: Minimum numeric grade considered by institution as "passing".
            Default value is 1.0, i.e. a "D" grade or better.
        min_num_credits_full_time: Minimum number of credits *attempted* per term
            for a student's enrollment intensity to be considered "full-time".
            Default value is 12.0.
        core_terms: Set of terms that together comprise the "core" of the academic year,
            in contrast with additional, usually shorter terms that may take place
            between core terms. Default value is {"FALL", "SPRING"}, which typically
            corresponds to a semester system; for schools on a trimester calendary,
            {"FALL", "WINTER", "SPRING"} is probably what you want.
        course_level_pattern
        peak_covid_terms
        key_course_subject_areas
        key_courses_ids

    References:
        - https://bigfuture.collegeboard.org/plan-for-college/get-started/how-to-convert-gpa-4.0-scale
    """
    first_term_of_year = infer_first_term_of_year(df_course["academic_term"])
    df_students = (
        df_cohort.pipe(standardize_cohort_dataset)
        .pipe(features.pdp.student.add_features, first_term_of_year=first_term_of_year)
    )  # fmt: skip
    df_courses_plus = (
        df_course.pipe(standardize_course_dataset)
        .pipe(
            features.pdp.course.add_features,
            min_passing_grade=min_passing_grade,
            course_level_pattern=course_level_pattern,
        )
        .pipe(
            features.pdp.term.add_features,
            first_term_of_year=first_term_of_year,
            core_terms=core_terms,  # type: ignore
            peak_covid_terms=peak_covid_terms,
        )
        .pipe(
            features.pdp.section.add_features,
            section_id_cols=["term_id", "course_id", "section_id"],
        )
    )
    df_student_terms = (
        features.pdp.student_term.aggregate_from_course_level_features(
            df_courses_plus,
            student_term_id_cols=["student_id", "term_id"],
            min_passing_grade=min_passing_grade,
            key_course_subject_areas=key_course_subject_areas,
            key_course_ids=key_course_ids,
        )
        .merge(df_students, how="inner", on=["institution_id", "student_id"])
        .pipe(
            features.pdp.student_term.add_features,
            min_num_credits_full_time=min_num_credits_full_time,
        )
    )
    df_student_terms_plus = (
        features.pdp.cumulative.add_features(
            df_student_terms,
            student_id_cols=["institution_id", "student_id"],
            sort_cols=["academic_year", "academic_term"],
        )
        # NOTE: it's important to standardize column names here to avoid name mismatches
        # when features are generated here (on-the-fly) as opposed to read (pre-computed)
        # from a delta table; spark can be configured to behave nicely...
        # but let's not take any chances
        .rename(columns=utils.misc.convert_to_snake_case)
    )
    return df_student_terms_plus


def standardize_cohort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop some columns from raw cohort dataset.

    Args:
        df: As output by :func:`dataio.read_raw_pdp_cohort_data_from_file()` .
    """
    LOGGER.info("standardizing cohort dataset ...")
    df_trf = (
        # drop redundant/useless/unwanted cols
        df.pipe(
            drop_columns_safely,
            cols=[
                # not a viable target variable, but highly correlated with it
                "time_to_credential",
                # not all demographics used for target variable bias checks
                "incarcerated_status",
                "military_status",
                "employment_status",
                "disability_status",
                "naspa_first_generation",
                # redundant
                "attendance_status_term_1",
                # covered indirectly by course dataset fields/features
                "gateway_math_status",
                "gateway_english_status",
                "attempted_gateway_math_year_1",
                "attempted_gateway_english_year_1",
                "completed_gateway_math_year_1",
                "completed_gateway_english_year_1",
                "gateway_math_grade_y_1",
                "gateway_english_grade_y_1",
                "attempted_dev_math_y_1",
                "attempted_dev_english_y_1",
                "completed_dev_math_y_1",
                "completed_dev_english_y_1",
                # let's assume we don't need other institution "demographics"
                "most_recent_bachelors_at_other_institution_state",
                "most_recent_associates_or_certificate_at_other_institution_state",
                "most_recent_last_enrollment_at_other_institution_state",
                "first_bachelors_at_other_institution_state",
                "first_associates_or_certificate_at_other_institution_state",
                "most_recent_bachelors_at_other_institution_carnegie",
                "most_recent_associates_or_certificate_at_other_institution_carnegie",
                "most_recent_last_enrollment_at_other_institution_carnegie",
                "first_bachelors_at_other_institution_carnegie",
                "first_associates_or_certificate_at_other_institution_carnegie",
                "most_recent_bachelors_at_other_institution_locale",
                "most_recent_associates_or_certificate_at_other_institution_locale",
                "most_recent_last_enrollment_at_other_institution_locale",
                "first_bachelors_at_other_institution_locale",
                "first_associates_or_certificate_at_other_institution_locale",
            ],
        )
        # as pdp adds more raw data columns, we'll want to ensure their presence here
        # so that feature generation code doesn't become a cascading mess of "if" checks
        .pipe(
            add_empty_cols_if_missing,
            col_val_dtypes={
                "years_to_latest_associates_at_cohort_inst": (None, "Int8"),
                "years_to_latest_certificate_at_cohort_inst": (None, "Int8"),
                "years_to_latest_associates_at_other_inst": (None, "Int8"),
                "years_to_latest_certificate_at_other_inst": (None, "Int8"),
                "first_year_to_associates_at_cohort_inst": (None, "Int8"),
                "first_year_to_certificate_at_cohort_inst": (None, "Int8"),
                "first_year_to_associates_at_other_inst": (None, "Int8"),
                "first_year_to_certificate_at_other_inst": (None, "Int8"),
            },
        )
    )
    return df_trf


def standardize_course_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop some columns and anomalous rows from raw course dataset.

    Args:
        df: As output by :func:`dataio.read_raw_pdp_course_data_from_file()` .
    """
    return (
        # drop rows for which we're missing key information
        df.pipe(drop_course_rows_missing_identifiers)
        # drop redundant/useless/unwanted cols
        .pipe(
            drop_columns_safely,
            cols=[
                # student demographics found in raw cohort dataset
                "cohort",
                "cohort_term",
                "student_age",
                "race",
                "ethnicity",
                "gender",
                # course name and aspects of core-ness not needed
                "course_name",
                "core_course_type",
                "core_competency_completed",
                "credential_engine_identifier",
                # enrollment record at other insts not needed
                "enrollment_record_at_other_institution_s_state_s",
                "enrollment_record_at_other_institution_s_carnegie_s",
                "enrollment_record_at_other_institution_s_locale_s",
            ],
        )
        # as pdp adds more raw data columns, we'll want to ensure their presence here
        # so that feature generation code doesn't become a cascading mess of "if" checks
        .pipe(
            add_empty_cols_if_missing,
            col_val_dtypes={"term_program_of_study": (None, "string")},
        )
    )


def add_empty_cols_if_missing(
    df: pd.DataFrame,
    *,
    col_val_dtypes: dict[
        str, tuple[t.Optional[str | bool | int | float | datetime.datetime], str]
    ],
) -> pd.DataFrame:
    """
    Add empty columns to ``df`` with names given by keys in ``col_val_dtypes``
    matched to values representing the null value and underlying dtype assigned to it
    -- provided the columns aren't already present in the dataframe.
    """
    return df.assign(
        **{
            col: pd.Series(data=val, index=df.index, dtype=dtype)
            for col, (val, dtype) in col_val_dtypes.items()
            if col not in df.columns
        }
    )


def clean_up_labeled_dataset_cols_and_vals(
    df: pd.DataFrame, num_credits_col: str = "cumsum_num_credits_earned"
) -> pd.DataFrame:
    """
    Drop a bunch of columns not needed or wanted for modeling, and set to null
    any values corresponding to time after a student's current year of enrollment.

    Args:
        df: DataFrame as created with features and targets and limited to the checkpoint term.
        num_credits_col: Name of the column containing cumulative earned credits.
    """
    num_credit_check = constants.DEFAULT_COURSE_CREDIT_CHECK
    credit_pattern = re.compile(rf"in_{num_credit_check}_creds")
    # To prevent data leakage, students that have not reached the 12 credits and not taken the course
    # by the checkpoint term (which this data is limited to at the time of this function),
    # will have the applicable in_12_credits columns set to null.
    for col in df.columns:
        if credit_pattern.search(col):
            df[col] = df[col].mask((df[num_credits_col] < num_credit_check))

    return (
        # drop many columns that *should not* be included as features in a model
        df.pipe(
            drop_columns_safely,
            cols=[
                # metadata
                "institution_id",
                "term_id",
                "academic_year",
                # "academic_term",  # keeping this to see if useful
                "cohort",
                # "cohort_term",  # keeping this to see if useful
                "cohort_id",
                "term_rank",
                "min_student_term_rank",
                "term_rank_core",
                "min_student_term_rank_core",
                "term_is_core",
                "term_rank_noncore",
                "min_student_term_rank_noncore",
                "term_is_noncore",
                # columns used to derive other features, but not features themselves
                # "grade",  # TODO: should this be course_grade?
                "course_ids",
                "course_subjects",
                "course_subject_areas",
                "min_student_term_rank",
                "min_student_term_rank_core",
                "min_student_term_rank_noncore",
                "sections_num_students_enrolled",
                "sections_num_students_passed",
                "sections_num_students_completed",
                "term_start_dt",
                "cohort_start_dt",
                "pell_status_first_year",
                # "outcome" variables / likely sources of data leakage
                "retention",
                "persistence",
                # years to bachelors
                "years_to_bachelors_at_cohort_inst",
                "years_to_bachelor_at_other_inst",
                "first_year_to_bachelors_at_cohort_inst",
                "first_year_to_bachelor_at_other_inst",
                # years to associates
                "years_to_latest_associates_at_cohort_inst",
                "years_to_latest_associates_at_other_inst",
                "first_year_to_associates_at_cohort_inst",
                "first_year_to_associates_at_other_inst",
                # years to associates / certificate
                "years_to_associates_or_certificate_at_cohort_inst",
                "years_to_associates_or_certificate_at_other_inst",
                "first_year_to_associates_or_certificate_at_cohort_inst",
                "first_year_to_associates_or_certificate_at_other_inst",
                # years of last enrollment
                "years_of_last_enrollment_at_cohort_institution",
                "years_of_last_enrollment_at_other_institution",
            ],
        ).assign(
            # keep "first year to X credential at Y inst"  or "years to latest X credential at Y inst" values if they occurred
            # in any year prior to the current year of enrollment; otherwise, set to null
            # in this case, the values themselves represent years
            **{
                col: ft.partial(_mask_year_values_based_on_enrollment_year, col=col)
                for col in df.columns[
                    df.columns.str.contains(
                        r"^(?:first_year_to_certificate|years_to_latest_certificate)"
                    )
                ].tolist()
            }
            # keep values in "*_year_X" columns if they occurred in any year prior
            # to the current year of enrollment; otherwise, set to null
            # in this case, the columns themselves represent years
            | {
                col: ft.partial(_mask_year_column_based_on_enrollment_year, col=col)
                for col in df.columns[df.columns.str.contains(r"_year_\d$")].tolist()
            }
            # do the same thing, only by term for term columns: "*_term_X"
            | {
                col: ft.partial(_mask_term_column_based_on_enrollment_term, col=col)
                for col in df.columns[df.columns.str.contains(r"_term_\d$")].tolist()
            }
        )
    )


def _mask_year_values_based_on_enrollment_year(
    df: pd.DataFrame,
    *,
    col: str,
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    return df[col].mask(df[col].ge(df[enrollment_year_col]), other=pd.NA)


def _mask_year_column_based_on_enrollment_year(
    df: pd.DataFrame,
    *,
    col: str,
    enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
) -> pd.Series:
    if match := re.search(r"_year_(?P<yr>\d)$", col):
        col_year = int(match.groupdict()["yr"])
    else:
        raise ValueError(f"column '{col}' does not end with '_year_NUM'")
    return df[col].mask(df[enrollment_year_col].le(col_year), other=pd.NA)


def _mask_term_column_based_on_enrollment_term(
    df: pd.DataFrame,
    *,
    col: str,
    enrollment_term_col: str = "cumnum_terms_enrolled",
) -> pd.Series:
    if match := re.search(r"_term_(?P<num>\d)$", col):
        col_term = int(match.groupdict()["num"])
    else:
        raise ValueError(f"column '{col}' does not end with '_term_NUM'")
    return df[col].mask(df[enrollment_term_col].lt(col_term), other=pd.NA)


def drop_course_rows_missing_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows from raw course dataset missing key course identifiers,
    specifically course prefix and number, which supposedly are partial records
    from students' enrollments at *other* institutions -- not wanted here!
    """
    num_rows_before = len(df)
    df = (
        df.loc[df[["course_prefix", "course_number"]].notna().all(axis=1), :]
        # reset range index so there are no gaps, which can cause subtle errors
        # when using `pd.DataFrame.assign()` to add features
        .reset_index(drop=True)
    )
    num_rows_after = len(df)
    if num_rows_after < num_rows_before:
        LOGGER.warning(
            "dropped %s rows from course dataset owing to missing identifiers",
            num_rows_before - num_rows_after,
        )
    return df


def drop_columns_safely(df: pd.DataFrame, *, cols: list[str]) -> pd.DataFrame:
    """
    Drop ``cols`` from ``df`` *safely*: If any are missing, log a clear warning,
    then drop the non-missing columns from the DataFrame without crashing.

    Args:
        df
        cols
    """
    df_cols = set(df.columns)
    drop_cols = set(cols) & df_cols
    if missing_cols := (set(cols) - df_cols):
        LOGGER.warning(
            "%s column%s not found in df: %s",
            len(missing_cols),
            "s" if len(missing_cols) > 1 else "",
            missing_cols,
        )
    df_trf = df.drop(columns=list(drop_cols))
    LOGGER.info("dropped %s columns safely", len(drop_cols))
    return df_trf


def infer_first_term_of_year(s: pd.Series) -> utils.types.TermType:
    """
    Infer the first term of the (academic) year by the ordering of its categorical values.

    See Also:
        - :class:`schemas.base.TermField()`
    """
    if isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered is True:
        first_term_of_year = s.cat.categories[0]
        LOGGER.info("'%s' inferred as the first term of the year", first_term_of_year)
        assert isinstance(first_term_of_year, str)  # type guard
        return first_term_of_year  # type: ignore
    else:
        raise ValueError(
            f"'{s.name}' series is not an ordered categorical: {s.dtype=} ..."
            "so the first term of the academic year can't be inferred. "
            "Update the raw course data schema to properly order its categories!"
        )


def infer_num_terms_in_year(s: pd.Series) -> int:
    """
    Infer the number of terms in the (academic) year by the number of its categorical values.

    See Also:
        - :class:`schemas.base.TermField()`
    """
    if isinstance(s.dtype, pd.CategoricalDtype):
        num_terms_in_year = len(s.cat.categories)
        LOGGER.info("%s inferred as the number of term in the year", num_terms_in_year)
        return num_terms_in_year
    else:
        raise ValueError(
            f"'{s.name}' series is not a categorical: {s.dtype=} ..."
            "so the number of term in the academic year can't be inferred. "
            "Update the raw course data schema to properly set its categories!"
        )
