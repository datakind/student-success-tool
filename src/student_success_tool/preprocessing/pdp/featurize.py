import logging
import typing as t

import pandas as pd

from ... import features, utils
from ...features.pdp import constants

LOGGER = logging.getLogger(__name__)


def featurize_student_terms(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    min_passing_grade: float = constants.DEFAULT_MIN_PASSING_GRADE,
    min_num_credits_full_time: float = constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
    course_level_pattern: str = constants.DEFAULT_COURSE_LEVEL_PATTERN,
    peak_covid_terms: set[tuple[str, str]] = constants.DEFAULT_PEAK_COVID_TERMS,
    key_course_subject_areas: t.Optional[list[str]] = None,
    key_course_ids: t.Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Make a student-term dataset from standardized cohort- and course-level datasets,
    including many features generated at the student-, course-, term-, section-,
    and student-term levels, as well as cumulatively over student-terms.

    Args:
        df_cohort: As output by :func:`standardize.standardize_cohort_dataset()` .
        df_course: As output by :func:`standardize.standardize_course_dataset()` .
        min_passing_grade: Minimum numeric grade considered by institution as "passing".
            Default value is 1.0, i.e. a "D" grade or better.
        min_num_credits_full_time: Minimum number of credits *attempted* per term
            for a student's enrollment intensity to be considered "full-time".
            Default value is 12.0.
        course_level_pattern
        peak_covid_terms
        key_course_subject_areas
        key_courses_ids

    References:
        - https://bigfuture.collegeboard.org/plan-for-college/get-started/how-to-convert-gpa-4.0-scale
    """
    first_term_of_year = infer_first_term_of_year(df_course["academic_term"])
    df_students = df_cohort.pipe(
        features.pdp.student.add_features,
        first_term_of_year=first_term_of_year,
    )
    df_courses_plus = (
        df_course.pipe(
            features.pdp.course.add_features,
            min_passing_grade=min_passing_grade,
            course_level_pattern=course_level_pattern,
        )
        .pipe(
            features.pdp.term.add_features,
            first_term_of_year=first_term_of_year,
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
