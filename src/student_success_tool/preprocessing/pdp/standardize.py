import datetime
import logging
import typing as t

import pandas as pd

LOGGER = logging.getLogger(__name__)


def standardize_cohort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unwanted columns from and add optional empty columns to the raw cohort dataset.

    Args:
        df: As output by :func:`dataio.pdp.read_raw_cohort_data()` .
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
    Drop unwanted columns from, add optional empty columns to, and remove anomalous rows
    from raw course dataset.

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
