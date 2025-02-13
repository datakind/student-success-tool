"""
HACK! This shows a concrete example of data preprocessing, encapsulated within a function.
Typically, this logic is implemented piecewise in the context of a Databricks notebook
(see: 01-preprocess-data-TEMPLATE.py).
"""

import functools as ft
import typing as t

import pandas as pd

from student_success_tool import modeling
from student_success_tool.analysis import pdp
from student_success_tool.configs.schemas import pdp_v2


def preprocess_data(
    df_course: pd.DataFrame,
    df_cohort: pd.DataFrame,
    *,
    run_type: t.Literal["predict", "train"],
    student_id_col: str = "student_guid",
    cfg: pdp_v2.PDPProjectConfigV2,
) -> pd.DataFrame:
    """
    Args:
        df_course
        df_cohort
        run_type
        student_id_col
        cfg
    """
    df_student_terms = pdp.dataops.make_student_term_dataset(
        df_cohort, df_course, **cfg.preprocessing.features.model_dump()
    )
    if run_type == "train":
        df_modeling = pdp.targets.failure_to_retain.make_labeled_dataset(
            df_student_terms, **cfg.preprocessing.target.params
        )
    else:
        eligible_students = pdp.targets.shared.select_students_by_criteria(
            df_student_terms,
            student_id_cols=student_id_col,
            **cfg.preprocessing.target.params["student_criteria"],
        )
        max_term_rank = df_student_terms["term_rank"].max()
        df_modeling = pd.merge(
            df_student_terms.loc[df_student_terms["term_rank"].eq(max_term_rank), :],
            eligible_students,
            on=student_id_col,
            how="inner",
        )
    df_modeling = pdp.dataops.clean_up_labeled_dataset_cols_and_vals(df_modeling)
    if run_type == "train":
        if cfg.split_col is not None and cfg.preprocessing.splits:
            df_modeling = df_modeling.assign(
                **{
                    cfg.split_col: ft.partial(
                        modeling.utils.compute_dataset_splits, seed=cfg.random_state
                    )
                }
            )
        if cfg.sample_weight_col is not None and cfg.preprocessing.sample_class_weight:
            df_modeling = df_modeling.assign(
                **{
                    cfg.sample_weight_col: ft.partial(
                        modeling.utils.compute_sample_weights,
                        target_col=cfg.target_col,
                        class_weight=cfg.preprocessing.sample_class_weight,
                    )
                }
            )
    return df_modeling
