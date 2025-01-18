import typing as t

import pydantic as pyd

from ...analysis.pdp import constants


class FeaturesConfig(pyd.BaseModel):
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


class TargetConfig(pyd.BaseModel):
    student_criteria: dict[str, object] = pyd.Field(
        default_factory=dict,
        description=(
            "Column name in modeling dataset mapped to one or more values that it must equal "
            "in order for the corresponding student to be considered 'eligible'. "
            "Multiple criteria are combined with a logical 'AND'."
        ),
    )
    # TODO: refine target functionality and expand on this configuration


class PreprocessingConfig(pyd.BaseModel):
    features: FeaturesConfig
    target: TargetConfig
    splits: dict[t.Literal["train", "test", "validate"], float] = pyd.Field(
        default={"train": 0.6, "test": 0.2, "validate": 0.2},
        description=(
            "Mapping of name to fraction of the full datset belonging to a given 'split', "
            "which is a randomized subset used for different parts of the modeling process"
        ),
    )
    sample_class_weight: t.Optional[t.Literal["balanced"] | dict[object, int]] = (
        pyd.Field(
            default=None,
            description=(
                "Weights associated with classes in the form ``{class_label: weight}`` "
                "or 'balanced' to automatically adjust weights inversely proportional "
                "to class frequencies in the input data. "
                "If null (default), then sample weights are not computed."
            ),
        )
    )

    @pyd.field_validator("splits", mode="after")
    @classmethod
    def check_split_fractions(cls, value: dict) -> dict:
        if (sum_fracs := sum(value.values())) != 1.0:
            raise pyd.ValidationError(
                f"split fractions must sum up to 1.0, but input sums up to {sum_fracs}"
            )
        return value


class FeatureSelectionConfig(pyd.BaseModel):
    """
    See Also:
        - :func:`modeling.feature_selection.select_features()`
    """

    non_feature_cols: t.Optional[list[str]] = None
    force_include_cols: t.Optional[list[str]] = None
    incomplete_threshold: float = 0.5
    low_variance_threshold: float = 0.0
    collinear_threshold: t.Optional[float] = 10.0


class TrainingConfig(pyd.BaseModel):
    """
    References:
        - https://docs.databricks.com/en/machine-learning/automl/automl-api-reference.html#classify
    """

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
    exclude_frameworks: t.Optional[list[str]] = pyd.Field(
        default=None,
        description="List of algorithm frameworks that AutoML excludes from training.",
    )
    primary_metric: str = pyd.Field(
        default="log_loss",
        description="Metric used to evaluate and rank model performance.",
    )
    timeout_minutes: t.Optional[int] = pyd.Field(
        default=None,
        description="Maximum time to wait for AutoML trials to complete.",
    )


class ModelingConfig(pyd.BaseModel):
    feature_selection: t.Optional[FeatureSelectionConfig] = None
    training: TrainingConfig


class InferenceConfig(pyd.BaseModel):
    num_top_features: int = pyd.Field(default=5)


class DatasetConfig(pyd.BaseModel):
    table_path: t.Optional[str] = pyd.Field(
        ...,
        description=(
            "Path to a table in Unity Catalog where dataset is stored, "
            "including the full three-level namespace: 'CATALOG.SCHEMA.TABLE'"
        ),
    )
    file_path: t.Optional[str] = pyd.Field(
        ...,
        description="Full, absolute path to dataset on disk, e.g. a Databricks Volume",
    )
    # TODO: if/when we allow different file formats, add this parameter ...
    # file_format: t.Optional[t.Literal["csv", "parquet"]] = pyd.Field(default=None)


class DatasetsConfig(pyd.BaseModel):
    raw: DatasetConfig
    preprocessed: t.Optional[DatasetConfig]
    predictions: t.Optional[DatasetConfig]


class TrainedModelConfig(pyd.BaseModel):
    experiment_id: str
    run_id: str
    model_type: t.Optional[t.Literal["sklearn", "xgboost", "lightgbm"]] = None
    min_prob_pos_label: t.Optional[float] = 0.5


class PDPProjectConfigV2(pyd.BaseModel):
    """Configuration (v2) schema for PDP SST projects."""

    institution_id: str
    institution_name: str

    # shared dataset parameters
    student_id_col: str = "student_guid"
    target_col: str = "target"
    split_col: str = "split"
    sample_weight_col: t.Optional[str] = None
    student_group_cols: t.Optional[list[str]] = pyd.Field(
        default=None,
        description=(
            "One or more column names in datasets containing student 'groups' "
            "to use for model bias assessment, but NOT as model features"
        ),
    )
    pos_label: t.Optional[int | bool | str] = True
    pred_col: str = "pred"
    pred_prob_col: str = "pred_prob"
    # other shared parameters
    random_state: t.Optional[int] = None

    labeled_dataset: DatasetsConfig
    trained_model: t.Optional[TrainedModelConfig] = None

    preprocessing: t.Optional[PreprocessingConfig] = None
    modeling: t.Optional[ModelingConfig] = None
    inference: t.Optional[InferenceConfig] = None

    # NOTE: this is for *pydantic* model -- not ML model -- configuration
    model_config = pyd.ConfigDict(extra="ignore", strict=True)
