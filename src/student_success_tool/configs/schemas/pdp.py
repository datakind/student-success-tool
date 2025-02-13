import typing as t

import pydantic as pyd

from ...analysis.pdp import constants


class PrepareModelingDatasetConfig(pyd.BaseModel):
    min_passing_grade: float = pyd.Field(
        default=constants.DEFAULT_MIN_PASSING_GRADE,
        description="Minimum numeric grade considered by institution as 'passing'",
        gt=0.0,
        lt=4.0,
    )
    min_num_credits_full_time: float = pyd.Field(
        default=constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
        description=(
            "Minimum number of credits *attempted* per term for a student's "
            "enrollment intensity to be considered 'full-time'."
        ),
        gt=0.0,
        lt=20.0,
    )
    course_level_pattern: str = pyd.Field(
        default=constants.DEFAULT_COURSE_LEVEL_PATTERN,
        description=(
            "Regular expression patttern that extracts a course's 'level' "
            "from a PDP course_number field"
        ),
    )
    peak_covid_terms: set[tuple[str, str]] = pyd.Field(
        default=constants.DEFAULT_PEAK_COVID_TERMS,
        description=(
            "Set of (academic year, academic term) pairs considered by institution "
            "as 'peak' COVID, for use in control variables to account for pandemic effects"
        ),
    )
    key_course_subject_areas: t.Optional[list[str]] = pyd.Field(
        default=None,
        description=(
            "One or more course subject areas (formatted as 2-digit CIP codes) "
            "for which custom features should be computed"
        ),
    )
    key_course_ids: t.Optional[list[str]] = pyd.Field(
        default=None,
        description=(
            "One or more course ids (formatted as '[COURSE_PREFIX][COURSE_NUMBER]') "
            "for which custom features should be computed"
        ),
    )
    target_student_criteria: dict[str, object] = pyd.Field(
        default_factory=dict,
        description=(
            "Column name in modeling dataset mapped to one or more values that it must equal "
            "in order for the corresponding student to be considered 'eligible'. "
            "Multiple criteria are combined with a logical 'AND'."
        ),
    )


class TrainEvaluateModelConfig(pyd.BaseModel):
    """
    References:
        - https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html#classify
    """

    dataset_table_path: str = pyd.Field(
        description=(
            "Path in Unity Catalog from which modeling dataset will be read, "
            "including the full three-level namespace: ``catalog.schema.table`` ."
        )
    )
    student_id_col: str = "student_guid"
    target_col: str = pyd.Field(
        default="target",
        description="Column name in dataset for the target label to be predicted.",
    )
    split_col: t.Optional[str] = pyd.Field(
        default=None,
        description="Column name in dataset for splitting into train/test/validate sets.",
    )
    sample_weight_col: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Column name in dataset that contains sample weights per row, "
            "typically used to adjust class importances for imbalanced data."
        ),
    )
    student_group_cols: t.Optional[list[str]] = pyd.Field(
        default=None,
        description=(
            "One or more column names in dataset containing student 'groups' "
            "to use for model bias assessment, but NOT as model features"
        ),
    )
    exclude_cols: t.Optional[list[str]] = pyd.Field(
        default=None,
        description="One or more column names in dataset to exclude from training.",
    )
    time_col: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Column name in dataset used to split train/test/validate sets chronologically, "
            "as an alternative to the randomized assignment in ``split_col`` ."
        ),
    )
    pos_label: t.Optional[int | bool | str] = pyd.Field(
        default=None,
        description=(
            "Positive class value in ``target_col`` "
            "(for binary classification problems only)."
        ),
    )
    exclude_frameworks: t.Optional[list[str]] = pyd.Field(
        default=None,
        description="List of algorithm frameworks that AutoML excludes from training.",
    )
    primary_metric: str = pyd.Field(
        default="f1", description="Metric used to evaluate and rank model performance."
    )
    timeout_minutes: t.Optional[int] = pyd.Field(
        default=None,
        description="Maximum time to wait for AutoML trials to complete.",
    )


class PDPProjectConfig(pyd.BaseModel):
    """
    Configuration schema for PDP SST projects, with top-level fields aligned to
    discrete steps in the SST data+modeling pipeline.
    """

    model_config = pyd.ConfigDict(extra="ignore", strict=True)

    institution_name: str
    # TODO: data_assessment_eda: DataAssessmentEDAConfig ?
    prepare_modeling_dataset: t.Optional[PrepareModelingDatasetConfig] = None
    train_evaluate_model: t.Optional[TrainEvaluateModelConfig] = None
