import logging
import typing as t

import pandas as pd

from . import constants, features

LOGGER = logging.getLogger(__name__)


def make_student_term_dataset(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    institution_state: str,
    min_passing_grade: str = constants.DEFAULT_MIN_PASSING_GRADE,
    course_level_pattern: str = constants.DEFAULT_COURSE_LEVEL_PATTERN,
    peak_covid_terms: set[tuple[str, str]] = constants.DEFAULT_PEAK_COVID_TERMS,
    num_terms_checkin: t.Optional[int] = None,
    key_course_subject_areas: t.Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Make a student-term dataset from raw cohort- and course-level datasets,
    including many features generated at the student-, course-, term-, section-,
    and student-term levels, as well as cumulatively over student-terms.

    Args:
        df_cohort: As output by :func:`dataio.read_raw_pdp_cohort_data_from_file()` .
        df_course: As output by :func:`dataio.read_raw_pdp_course_data_from_file()` .
        institution_state
        min_passing_grade: Minimum numeric grade considered by institution as "passing".
        course_level_pattern
        peak_covid_terms
        num_terms_checkin: If known and applicable, the fixed check-in term from which
            model predictions will be made, specified as an integer, where
            1 => check-in is after the student's first term, 2 => second term, etc.
            If 1, all columns only known after the first year will be dropped
            to prevent data leakage.
        key_course_subject_areas

    TODO: Get rid of num_terms_checkin plus associated logic, maybe?
    """
    df_students = (
        df_cohort.pipe(standardize_cohort_dataset, num_terms_checkin=num_terms_checkin)
        .pipe(features.student.add_features, institution_state=institution_state)
    )  # fmt: skip
    df_courses_plus = (
        df_course.pipe(standardize_course_dataset)
        .pipe(
            features.course.add_features,
            min_passing_grade=min_passing_grade,
            course_level_pattern=course_level_pattern,
        )
        .pipe(features.term.add_features, peak_covid_terms=peak_covid_terms)
        .pipe(
            features.section.add_features,
            section_id_cols=["term_id", "course_id", "section_id"],
        )
    )
    df_student_terms = (
        features.student_term.aggregate_from_course_level_features(
            df_courses_plus,
            student_term_id_cols=["student_guid", "term_id"],
            key_course_subject_areas=key_course_subject_areas,
        )
        .merge(df_students, how="inner", on=["institution_id", "student_guid"])
        .pipe(features.student_term.add_features)
    )
    df_student_terms_plus = features.cumulative.add_features(
        df_student_terms,
        student_id_cols=["institution_id", "student_guid"],
        sort_cols=["academic_year", "academic_term"],
    )
    return df_student_terms_plus


def standardize_cohort_dataset(
    df: pd.DataFrame, *, num_terms_checkin: t.Optional[int] = None
) -> pd.DataFrame:
    """
    Drop some columns from raw cohort dataset.

    Args:
        df: As output by :func:`dataio.read_raw_pdp_cohort_data_from_file()` .
        num_terms_checkin: If known and applicable, the fixed check-in term from which
            model predictions will be made, specified as an integer, where
            1 => check-in is after the student's first term, 2 => second term, etc.
            If 1, all columns only known after the first year will be dropped
            to prevent data leakage.
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
                # derived directly from course dataset
                "number_of_credits_attempted_year_1",
                "number_of_credits_attempted_year_2",
                "number_of_credits_attempted_year_3",
                "number_of_credits_attempted_year_4",
                "number_of_credits_earned_year_1",
                "number_of_credits_earned_year_2",
                "number_of_credits_earned_year_3",
                "number_of_credits_earned_year_4",
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
                # "most_recent_last_enrollment_at_other_institution_state",
                "first_bachelors_at_other_institution_state",
                "first_associates_or_certificate_at_other_institution_state",
                "most_recent_bachelors_at_other_institution_carnegie",
                "most_recent_associates_or_certificate_at_other_institution_carnegie",
                # "most_recent_last_enrollment_at_other_institution_carnegie",
                "first_bachelors_at_other_institution_carnegie",
                "first_associates_or_certificate_at_other_institution_carnegie",
                "most_recent_bachelors_at_other_institution_locale",
                "most_recent_associates_or_certificate_at_other_institution_locale",
                # "most_recent_last_enrollment_at_other_institution_locale",
                "first_bachelors_at_other_institution_locale",
                "first_associates_or_certificate_at_other_institution_locale",
            ],
        )
    )
    if num_terms_checkin is not None and num_terms_checkin < 2:
        LOGGER.info("num_terms_checkin=1, so dropping additional year 1+ columns ...")
        df_trf = df_trf.pipe(
            drop_columns_safely,
            cols=[
                "gpa_group_year_1",
                "program_of_study_year_1",
                "retention",
                "persistence",
            ],
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
    )


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


def infer_first_term_of_year(s: pd.Series) -> str:
    """
    Infer the first term of the (academic) year by the ordering of its categorical values.

    See Also:
        - :class:`schemas.base.TermField()`
    """
    if isinstance(s.dtype, pd.CategoricalDtype) and s.cat.ordered is True:
        first_term_of_year = s.cat.categories[0]
        LOGGER.info("'%s' inferred as the first term of the year", first_term_of_year)
        assert isinstance(first_term_of_year, str)  # type guard
        return first_term_of_year
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