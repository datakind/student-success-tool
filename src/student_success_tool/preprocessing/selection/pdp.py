import logging
from collections.abc import Collection

import numpy as np
import pandas as pd

from ... import utils

LOGGER = logging.getLogger(__name__)


def select_students_by_attributes(
    df: pd.DataFrame,
    *,
    student_id_cols: str | list[str] = "student_id",
    **criteria: object | Collection[object],
) -> pd.DataFrame:
    """
    Select distinct students in ``df`` with specified attributes,
    according to the logical "AND" of one or more selection criteria.

    Args:
        df: Student-term dataset.
        student_id_cols: Column(s) that uniquely identify students in ``df`` ,
            used to drop duplicates.
        **criteria: Column name in ``df`` mapped to one or multiple values
            that it must equal in order for the corresponding student to be selected.
            Multiple criteria are combined with a logical "AND". For example,
            ``enrollment_type="FIRST-TIME", enrollment_intensity_first_term=["FULL-TIME", "PART-TIME"]``
            selects first-time students who enroll at full- or part-time intensity.

    Warning:
        This assumes that students' attribute values are the same across all terms.
        If that assumption is violated, this code actually checks if the student
        satisfied all of the criteria during _any_ of their terms.
    """
    if not criteria:
        raise ValueError("one or more selection criteria must be specified")

    student_id_cols = utils.types.to_list(student_id_cols)
    nuq_students_in = df.groupby(by=student_id_cols, sort=False).ngroups
    is_selecteds = [
        df[key].isin(set(val))  # type: ignore
        if utils.types.is_collection_but_not_string(val)
        else df[key].eq(val).fillna(value=False)  # type: ignore
        for key, val in criteria.items()
    ]
    for (key, val), is_selected in zip(criteria.items(), is_selecteds):
        nuq_students_criterion = (
            df.loc[is_selected, student_id_cols]
            .groupby(by=student_id_cols, sort=False)
            .ngroups
        )
        _log_selection(
            nuq_students_in, nuq_students_criterion, f"{key}={val} criterion ..."
        )
    is_selected = np.logical_and.reduce(is_selecteds)
    df_selected = (
        df.loc[is_selected, student_id_cols + list(criteria.keys())]
        # df is at student-term level; get student ids from first selected terms only
        .drop_duplicates(subset=student_id_cols, ignore_index=True)
        .set_index(student_id_cols)
    )
    _log_selection(nuq_students_in, len(df_selected), "all criteria")
    return df_selected


def _log_selection(
    nunique_students_in: int, nunique_students_out: int, case: str
) -> None:
    LOGGER.info(
        "%s out of %s (%s%%) students selected by %s",
        nunique_students_out,
        nunique_students_in,
        round(100 * nunique_students_out / nunique_students_in, 1),
        case,
    )
