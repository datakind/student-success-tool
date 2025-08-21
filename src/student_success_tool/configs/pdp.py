import re
import typing as t

import pydantic as pyd

# TODO: set field defaults using literals here instead?
from ..preprocessing.features.pdp import constants
from ..utils import types


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
    datasets: "AllDatasetStagesConfig" = pyd.Field(
        description=(
            "Key datasets produced by the pipeline represented here in this config"
        ),
    )
    model: t.Optional["ModelConfig"] = pyd.Field(
        default=None,
        description=(
            "Essential metadata for identifying and loading trained model artifacts, "
            "as produced by the pipeline represented here in this config"
        ),
    )

    # key steps in project pipeline
    preprocessing: t.Optional["PreprocessingConfig"] = None
    modeling: t.Optional["ModelingConfig"] = None
    inference: t.Optional["InferenceConfig"] = None

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
    model_config = pyd.ConfigDict(extra="forbid", strict=True)


class BronzeDatasetConfig(pyd.BaseModel):
    raw_course: "DatasetIOConfig"
    raw_cohort: "DatasetIOConfig"


class SilverDatasetConfig(pyd.BaseModel):
    preprocessed: "DatasetIOConfig"
    modeling: "DatasetIOConfig"


class GoldDatasetConfig(pyd.BaseModel):
    advisor_output: "DatasetIOConfig"


class AllDatasetStagesConfig(pyd.BaseModel):
    bronze: BronzeDatasetConfig
    silver: SilverDatasetConfig
    gold: GoldDatasetConfig


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


class ModelConfig(pyd.BaseModel):
    experiment_id: str
    run_id: str
    framework: t.Optional[t.Literal["sklearn", "xgboost", "lightgbm"]] = None

    @pyd.computed_field  # type: ignore[misc]
    @property
    def mlflow_model_uri(self) -> str:
        return f"runs:/{self.run_id}/model"


class PreprocessingConfig(pyd.BaseModel):
    features: "FeaturesConfig"
    selection: "SelectionConfig"
    checkpoint: "CheckpointNthConfig | CheckpointFirstConfig | CheckpointLastConfig | CheckpointFirstAtNumCreditsEarnedConfig | CheckpointFirstWithinCohortConfig | CheckpointLastInEnrollmentYearConfig"
    target: "TargetGraduationConfig | TargetRetentionConfig | TargetCreditsEarnedConfig"
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
    include_pre_cohort_courses: bool = pyd.Field(
        default=False,
        description=(
            "Whether to include course records that occurred before the student's cohort term. Usually, we do end up excluding these so the default will always be False unless set otherwise."
        ),
    )

    @pyd.field_validator("splits", mode="after")
    @classmethod
    def check_split_fractions(cls, value: dict) -> dict:
        if (sum_fracs := sum(value.values())) != 1.0:
            raise pyd.ValidationError(
                f"split fractions must sum up to 1.0, but input sums up to {sum_fracs}"
            )
        return value


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
    key_course_subject_areas: t.Optional[t.List[t.Union[str, t.List[str]]]] = pyd.Field(
        default=None,
        description=(
            "One or more course subject areas (formatted as 2-digit CIP codes) "
            "for which custom features should be computed, can be a list or include a nested list"
            "Example: ['51', ['27', '48']], so you would get features for 51 alone and features for 27 and 48 combined."
        ),
    )
    key_course_ids: t.Optional[t.List[t.Union[str, t.List[str]]]] = pyd.Field(
        default=None,
        description=(
            "One or more course ids (formatted as '[COURSE_PREFIX][COURSE_NUMBER]') "
            "for which custom features should be computed, can be a list or include nested lists"
        ),
    )


class SelectionConfig(pyd.BaseModel):
    student_criteria: dict[str, object] = pyd.Field(
        default_factory=dict,
        description=(
            "Column name in modeling dataset mapped to one or more values that it must equal "
            "in order for the corresponding student to be considered 'eligible'. "
            "Multiple criteria are combined with a logical 'AND'."
        ),
    )
    intensity_time_limits: t.Optional[types.IntensityTimeLimitsType] = pyd.Field(
        default=None,
        description=(
            "Mapping of enrollment intensity value (e.g. 'FULL-TIME') to the max number "
            "years or terms (e.g. [4.0, 'year'], [12.0, 'term']) considered to be 'on-time' "
            "for a given target (e.g. graduation, credits earned), "
            "where the numeric values are for the time between 'checkpoint' and 'target' "
            "terms. Passing special '*' as the only key applies the corresponding time limits "
            "to all students, regardless of intensity."
        ),
    )
    params: dict[str, object] = pyd.Field(
        default_factory=dict,
        description="Any additional parameters needed to configure student selection",
    )


class CheckpointBaseConfig(pyd.BaseModel):
    name: str = pyd.Field(
        default=...,
        description="Descriptive name for checkpoint, used as a component in model name",
    )
    type_: types.CheckpointTypeType = pyd.Field(
        default=..., description="Type of checkpoint to which config is applied"
    )
    sort_cols: str | list[str] = pyd.Field(
        default="term_rank",
        description="Column(s) used to sort students' terms, typically chronologically.",
    )
    include_cols: t.Optional[list[str]] = pyd.Field(
        default=None,
        description="Optional subset of columns to include in checkpoint student-terms.",
    )


class CheckpointNthConfig(CheckpointBaseConfig):
    type_: types.CheckpointTypeType = "nth"
    n: int = pyd.Field(default=...)
    term_is_pre_cohort_col: t.Optional[str] = pyd.Field(default="term_is_pre_cohort")
    exclude_pre_cohort_terms: t.Optional[bool] = pyd.Field(default=True)
    term_is_core_col: t.Optional[str] = pyd.Field(default="term_is_core")
    exclude_non_core_terms: t.Optional[bool] = pyd.Field(default=True)
    enrollment_year_col: t.Optional[str] = pyd.Field(default=None)
    valid_enrollment_year: t.Optional[int] = pyd.Field(default=None)


class CheckpointFirstConfig(CheckpointBaseConfig):
    type_: types.CheckpointTypeType = "first"


class CheckpointLastConfig(CheckpointBaseConfig):
    type_: types.CheckpointTypeType = "last"


class CheckpointFirstAtNumCreditsEarnedConfig(CheckpointBaseConfig):
    type_: types.CheckpointTypeType = "first_at_num_credits_earned"
    min_num_credits: float = pyd.Field(default=...)
    num_credits_col: str = pyd.Field(default="num_credits_earned_cumsum")


class CheckpointFirstWithinCohortConfig(CheckpointBaseConfig):
    type_: types.CheckpointTypeType = "first_within_cohort"
    term_is_pre_cohort_col: str = pyd.Field(default="term_is_pre_cohort")


class CheckpointLastInEnrollmentYearConfig(CheckpointBaseConfig):
    type_: types.CheckpointTypeType = "last_in_enrollment_year"
    enrollment_year: float = pyd.Field(default=...)
    enrollment_year_col: str = pyd.Field(default="year_of_enrollment_at_cohort_inst")


class TargetBaseConfig(pyd.BaseModel):
    name: str = pyd.Field(
        default=...,
        description="Descriptive name for target, used as a component in model name",
    )
    type_: types.TargetTypeType = pyd.Field(
        default=..., description="Type of target to which config is applied"
    )


class TargetGraduationConfig(TargetBaseConfig):
    type_: types.TargetTypeType = "graduation"
    intensity_time_limits: types.IntensityTimeLimitsType
    years_to_degree_col: str
    num_terms_in_year: int = pyd.Field(default=4)
    max_term_rank: int | t.Literal["infer"]


class TargetRetentionConfig(TargetBaseConfig):
    type_: types.TargetTypeType = "retention"
    max_academic_year: str | t.Literal["infer"]


class TargetCreditsEarnedConfig(TargetBaseConfig):
    type_: types.TargetTypeType = "credits_earned"
    min_num_credits: float
    # TODO: is there any way to represent checkpoint arg in toml, given its dtype?
    intensity_time_limits: types.IntensityTimeLimitsType
    num_terms_in_year: int = pyd.Field(default=4)
    max_term_rank: int | t.Literal["infer"]


class ModelingConfig(pyd.BaseModel):
    feature_selection: t.Optional["FeatureSelectionConfig"] = None
    training: "TrainingConfig"
    evaluation: t.Optional["EvaluationConfig"] = None


class FeatureSelectionConfig(pyd.BaseModel):
    """
    See Also:
        - :func:`modeling.feature_selection.select_features()`
    """

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


class InferenceConfig(pyd.BaseModel):
    num_top_features: int = pyd.Field(default=5)
    min_prob_pos_label: t.Optional[float] = 0.5
    # TODO: extend this configuration, maybe?
