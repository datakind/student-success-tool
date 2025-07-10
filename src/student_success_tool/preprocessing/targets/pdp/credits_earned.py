import typing as t

import numpy as np
import pandas as pd

from .... import utils
from ... import checkpoints
from . import shared


def compute_target(
    df: pd.DataFrame,
    *,
    min_num_credits: float,
    checkpoint: pd.DataFrame | t.Callable[[pd.DataFrame], pd.DataFrame],
    intensity_time_limits: utils.types.IntensityTimeLimitsType,
    num_terms_in_year: int = 4,
    max_term_rank: int | t.Literal["infer"] = "infer",
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "student_term_enrollment_intensity",
    num_credits_col: str = "cumsum_num_credits_earned",
    term_rank_col: str = "term_rank",
) -> pd.Series:
    """
    Compute *insufficient* credits earned target for each distinct student in ``df`` ,
    for which intensity-specific time limits determine if credits earned is "on-time".

    Args:
        df: Student-term dataset.
        min_num_credits: Minimum number of credits earned within specified time limits
            to be considered a *success* => target=False.
        checkpoint: "Checkpoint" from which time limits to target term are determined,
            typically either the first enrolled term or the first term above an intermediate
            number of credits earned; may be given as a data frame with one row per student,
            or as a callable that takes ``df`` as input and returns all checkpoint terms.
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms considered to be "on-time" for
            the target number of credits earned (e.g. [4.0, "year"], [12.0, "term"]),
            where the numeric values are for the time between "checkpoint" and "target"
            terms. Passing special "*" as the only key applies the corresponding time limits
            to all students, regardless of intensity.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert from year-based time limits to term-based time limits;
            default value assumes FALL, WINTER, SPRING, and SUMMER terms.
        max_term_rank: Maximum term rank value in the full dataset ``df`` , either inferred
            from ``df[term_rank_col]`` itself or as a manually specified value which
            may be different from the actual max value in ``df`` , depending on use case.
        student_id_cols: Columns that uniquely identify students, used for grouping rows.
        enrollment_intensity_col: Column whose values give students' "enrollment intensity"
            (usually either "FULL-TIME" or "PART-TIME"), for which the most common
            value per student is used when comparing against intensity-specific time limits.
        num_credits_col
        term_rank_col: Column whose values give the absolute integer ranking of a given
            term within the full dataset ``df`` .
    """
    student_id_cols = utils.types.to_list(student_id_cols)
    # we want a target for every student in input df; this will ensure it
    df_distinct_students = df[student_id_cols].drop_duplicates(ignore_index=True)
    df_ckpt = (
        checkpoint.copy(deep=True)
        if isinstance(checkpoint, pd.DataFrame)
        else checkpoint(df)
    )
    if df_ckpt.groupby(by=student_id_cols).size().gt(1).any():
        raise ValueError("checkpoint df must include exactly 1 row per student")

    df_tgt = checkpoints.pdp.first_student_terms_at_num_credits_earned(
        df,
        min_num_credits=min_num_credits,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        num_credits_col=num_credits_col,
        include_cols=[enrollment_intensity_col],
    )
    df_at = pd.merge(
        df_ckpt,
        df_tgt,
        on=student_id_cols,
        how="left",
        suffixes=("_ckpt", "_tgt"),
    )
    # convert from year limits to term limits, as needed
    intensity_num_terms = utils.misc.convert_intensity_time_limits(
        "term", intensity_time_limits, num_terms_in_year=num_terms_in_year
    )
    # compute all intensity/term boolean arrays separately
    # then combine with a logical OR
    tr_col = term_rank_col  # hack, so logic below fits on lines
    targets = [
        (
            # enrollment intensity is equal to a specified value or "*" given as intensity
            (
                df_at[f"{enrollment_intensity_col}_ckpt"].eq(intensity)
                | (intensity == "*")
            )
            & (
                # num terms between target/checkpoint greater than max num allowed
                (df_at[f"{tr_col}_tgt"] - df_at[f"{tr_col}_ckpt"]).gt(num_terms)
                # or they *never* earned enough credits for target
                | df_at[f"{tr_col}_tgt"].isna()
            )
        )
        for intensity, num_terms in intensity_num_terms.items()
    ]
    target = np.logical_or.reduce(targets)
    # assign True to all students passing intensity/year condition(s) above
    df_target_true = (
        df_at.loc[target, student_id_cols]
        .assign(target=True)
        .astype({"target": "boolean"})
    )
    # get all students for which a target label can accurately be computed
    # i.e. the data in df covers their last "on-time" graduation term
    df_labelable_students = shared.get_students_with_max_target_term_in_dataset(
        df,
        checkpoint=df_ckpt,
        intensity_time_limits=intensity_time_limits,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        student_id_cols=student_id_cols,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    df_labeled = (
        # match positive labels to label-able students
        pd.merge(df_labelable_students, df_target_true, on=student_id_cols, how="left")
        # assign False to all label-able students not already assigned True
        .fillna({"target": False})
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    df_all_student_targets = (
        # assign null target to all students
        df_distinct_students.assign(target=pd.Series(pd.NA, dtype="boolean"))
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    # update null targets in-place with bool targets on matching student-id indexes
    df_all_student_targets.update(df_labeled)
    # #drop if target is uncalculable (null)
    df_all_student_targets["target"] = (
        df_all_student_targets["target"].astype("boolean").dropna()
    )
    # return as a series with target as values and student ids as index
    return df_all_student_targets.loc[:, "target"]
