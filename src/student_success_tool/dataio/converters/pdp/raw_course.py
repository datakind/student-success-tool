import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def dedupe_by_renumbering_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate rows in raw course data ``df`` by renumbering courses, such that
    the data passes data schema uniqueness requirements.

    Args:
        df: Raw course dataset

    Warning:
        This logic assumes that all rows are actually valid, and that the school's
        course numbering is wonky. This can happen if, for example, supplemental courses
        (e.g. labs) for a main course share the same course number.

        Don't use this function if there are actual duplicate records in the data!

    See Also:
        :class:`schemas.pdp.RawPDPCourseDataSchema`
    """
    # HACK: infer the correct student id col in raw data from the data itself
    student_id_col = (
        "student_guid"
        if "student_guid" in df.columns
        else "study_id"
        if "study_id" in df.columns
        else "student_id"
    )
    unique_cols = [
        student_id_col,
        "academic_year",
        "academic_term",
        "course_prefix",
        "course_number",
        "section_id",
    ]
    deduped_course_numbers = (
        # get all duplicated records
        df.loc[df.duplicated(unique_cols, keep=False), :]
        # sort in descending order of num credits attempted, which assumes that
        # courses worth more credits are "primary" and should be numbered first
        .sort_values(
            by=unique_cols + ["number_of_credits_attempted"],
            ascending=False,
            ignore_index=False,
        )
        .assign(
            grp_num=(
                lambda df: df.groupby(unique_cols)["course_number"].transform(
                    "cumcount"
                )
                + 1
            ),
            # add "group number" suffix to course numbers in order to disambiguate
            course_number=lambda df: df["course_number"].str.cat(
                df["grp_num"].astype("string"), sep="-"
            ),
        )
        .loc[:, ["course_number"]]
    )
    LOGGER.warning(
        "%s duplicate course records found; course numbers modified to avoid duplicates",
        len(deduped_course_numbers),
    )
    # update course numbers for dupe'd records in-place
    df.update(deduped_course_numbers, overwrite=True)
    return df
