import re
import typing as t

import pydantic as pyd

from ...features.pdp import constants


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
    core_terms: set[str] = pyd.Field(
        default=constants.DEFAULT_CORE_TERMS,
        description=(
            "Set of terms that together comprise the 'core' of the academic year, "
            "in contrast with additional, usually shorter terms that may take place "
            "between core terms"
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
    params: dict[str, object] = pyd.Field(default_factory=dict)


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


class EvaluationConfig(pyd.BaseModel):
    topn_runs_included: int = pyd.Field(
        default=3,
        description="Number of top-scoring mlflow runs to include for detailed evaluation",
    )


class ModelingConfig(pyd.BaseModel):
    feature_selection: t.Optional[FeatureSelectionConfig] = None
    training: TrainingConfig
    evaluation: t.Optional[EvaluationConfig] = None


class InferenceConfig(pyd.BaseModel):
    num_top_features: int = pyd.Field(default=5)
    min_prob_pos_label: t.Optional[float] = 0.5
    # TODO: extend this configuration, maybe?


class DatasetIOConfig(pyd.BaseModel):
    table_path: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Path to a table in Unity Catalog where dataset is stored, "
            "including the full three-level namespace: 'CATALOG.SCHEMA.TABLE'"
        ),
    )
    file_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Full, absolute path to dataset on disk, e.g. a Databricks Volume",
    )
    # TODO: if/when we allow different file formats, add this parameter ...
    # file_format: t.Optional[t.Literal["csv", "parquet"]] = pyd.Field(default=None)

    @pyd.model_validator(mode="after")
    def check_some_nonnull_inputs(self):
        if self.table_path is None and self.file_path is None:
            raise ValueError("table_path and/or file_path must be non-null")
        return self


class DatasetConfig(pyd.BaseModel):
    raw_course: DatasetIOConfig
    raw_cohort: DatasetIOConfig
    preprocessed: t.Optional[DatasetIOConfig] = None
    predictions: t.Optional[DatasetIOConfig] = None
    finalized: t.Optional[DatasetIOConfig] = None


class ModelConfig(pyd.BaseModel):
    experiment_id: str
    run_id: str
    framework: t.Optional[t.Literal["sklearn", "xgboost", "lightgbm"]] = None

    @pyd.computed_field  # type: ignore[misc]
    @property
    def mlflow_model_uri(self) -> str:
        return f"runs:/{self.run_id}/model"


class PDPProjectConfig(pyd.BaseModel):
    """Configuration schema for SST PDP projects."""

    institution_id: str = pyd.Field(
        ...,
        description=(
            "Unique (ASCII-only) identifier for institution; used in naming things "
            "such as source directories, catalog schemas, keys in shared configs, etc."
        ),
    )
    institution_name: str = pyd.Field(
        ...,
        description=(
            "Readable 'display' name for institution, distinct from the 'id'; "
            "probably just the school's 'official', public name"
        ),
    )

    # shared parameters
    student_id_col: str = "student_id"
    target_col: str = "target"
    split_col: t.Optional[str] = "split"
    sample_weight_col: t.Optional[str] = "sample_weight"
    student_group_cols: t.Optional[list[str]] = pyd.Field(
        default=["student_age", "race", "ethnicity", "gender", "first_gen"],
        description=(
            "One or more column names in datasets containing student 'groups' "
            "to use for model bias assessment, but *not* as model features"
        ),
    )
    pred_col: str = "pred"
    pred_prob_col: str = "pred_prob"
    pos_label: t.Optional[int | bool | str] = True
    random_state: t.Optional[int] = None

    # key artifacts produced by project pipeline
    datasets: dict[str, DatasetConfig] = pyd.Field(
        default={},
        description=(
            "Mapping of dataset name, e.g. 'labeled', to file/table paths for each "
            "derived form produced by steps in the data transformation pipeline, "
            "used to load the artifacts from storage"
        ),
    )
    models: dict[str, ModelConfig] = pyd.Field(
        default={},
        description=(
            "Mapping of model name, e.g. 'graduation', to MLFlow metadata used to "
            "load the trained artifact from storage"
        ),
    )
    # key steps in project pipeline
    preprocessing: t.Optional[PreprocessingConfig] = None
    modeling: t.Optional[ModelingConfig] = None
    inference: t.Optional[InferenceConfig] = None

    @pyd.computed_field  # type: ignore[misc]
    @property
    def non_feature_cols(self) -> list[str]:
        return (
            [self.student_id_col, self.target_col]
            + ([self.split_col] if self.split_col else [])
            + ([self.sample_weight_col] if self.sample_weight_col else [])
            + (self.student_group_cols or [])
        )

    @pyd.field_validator("institution_id", mode="after")
    @classmethod
    def check_institution_id_isascii(cls, value: str) -> str:
        if not re.search(r"^\w+$", value, flags=re.ASCII):
            raise ValueError(f"institution_id='{value}' is not ASCII-only")
        return value

    # NOTE: this is for *pydantic* model -- not ML model -- configuration
    model_config = pyd.ConfigDict(extra="ignore", strict=True)
