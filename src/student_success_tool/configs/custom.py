import re
import typing as t

import pydantic as pyd

# allowed primary metrics by framework
_ALLOWED_BY_FRAMEWORK = {
    "sklearn": {"f1", "log_loss", "precision", "accuracy", "roc_auc"},
    "h2o": {"logloss", "auc", "aucpr", "rmse", "mae", "mean_per_class_error"},
}


class CustomProjectConfig(pyd.BaseModel):
    """Configuration schema for SST Custom projects."""

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
    split_col: str = "split"
    sample_weight_col: t.Optional[str] = "sample_weight"
    student_group_cols: t.Optional[list[str]] = pyd.Field(
        default=["age_group", "race", "gender"],
        description=(
            "One or more column names in datasets containing student 'groups' "
            "to use for model bias assessment, but *not* as model features"
        ),
    )
    student_group_aliases: dict[str, str] = pyd.Field(
        default_factory=dict,
        description=(
            "Mapping from raw column name (e.g., GENDER_DESC) to "
            "friendly label (e.g., 'Gender') for use in model cards"
        ),
    )
    pred_col: str = "pred"
    pred_prob_col: str = "pred_prob"
    pos_label: t.Optional[int | bool | str] = True
    random_state: t.Optional[int] = 12345

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

    @pyd.model_validator(mode="after")
    def check_sample_weight_requires_random_state(self):
        if self.sample_weight_col and self.random_state is None:
            raise ValueError(
                "random_state must be specified if sample_weight_col is provided"
            )
        return self

    @pyd.model_validator(mode="after")
    def validate_student_group_aliases(self) -> "CustomProjectConfig":
        missing = [
            col
            for col in (self.student_group_cols or [])
            if col not in (self.student_group_aliases or {})
        ]
        if missing:
            raise ValueError(f"Missing student_group_aliases for: {missing}")
        return self

    @pyd.model_validator(mode="after")
    def _normalize_and_validate_metric_using_framework(self) -> "CustomProjectConfig":
        fw = "sklearn"
        if self.model and self.model.framework:
            fw = self.model.framework

        if (
            self.modeling
            and self.modeling.training
            and self.modeling.training.primary_metric
        ):
            pm = self.modeling.training.primary_metric
            allowed = _ALLOWED_BY_FRAMEWORK.get(fw)
            if not allowed:
                raise ValueError(f"Unknown framework '{fw}' for metric normalization")

            # choose framework-preferred spelling for log loss
            if pm in {"logloss", "log_loss"}:
                pm = "logloss" if fw == "h2o" else "log_loss"

            # final gate
            if pm not in allowed:
                raise ValueError(
                    f"Unsupported primary_metric '{pm}' for framework '{fw}'. "
                    f"Allowed: {sorted(allowed)}"
                )
            self.modeling.training.primary_metric = pm
        return self

    @pyd.model_validator(mode="after")
    def _validate_h2o_inference_background_sample(self) -> "CustomProjectConfig":
        # Default to sklearn if framework is unset (keeps STANDARD behavior)
        fw = "sklearn"
        if self.model and self.model.framework:
            fw = self.model.framework

        # Only enforce for H2O
        if fw != "h2o":
            return self

        if self.inference and self.inference.background_data_sample is not None:
            n = self.inference.background_data_sample
            if not (500 <= n <= 2000):
                raise ValueError(
                    "For H2O, inference.background_data_sample must be between 500 and 2000."
                )
        return self


class DatasetConfig(pyd.BaseModel):
    train_file_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Absolute path to training dataset on disk.",
    )
    predict_file_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Absolute path to prediction/inference dataset on disk.",
    )
    train_table_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Unity Catalog table path for training dataset, e.g., 'catalog.schema.table'.",
    )
    predict_table_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Unity Catalog table path for prediction/inference dataset.",
    )
    file_path: t.Optional[str] = None
    table_path: t.Optional[str] = None

    primary_keys: t.Optional[t.List[str]] = pyd.Field(
        default=None,
        description="Primary keys utilized for data validation, if applicable",
    )
    drop_cols: t.Optional[t.List[str]] = pyd.Field(
        default=None,
        description="Columns to be dropped during pre-processing, if applicable",
    )
    non_null_cols: t.Optional[t.List[str]] = pyd.Field(
        default=None,
        description="Columns to be validated as non-null, if applicable",
    )

    @pyd.model_validator(mode="after")
    def validate_paths(self) -> "DatasetConfig":
        any_paths = [
            self.train_file_path,
            self.predict_file_path,
            self.train_table_path,
            self.predict_table_path,
            self.file_path,  # Legacy, not used in pipeline/DB workflow
            self.table_path,  # Legacy, not used in pipeline/DB workflow
        ]
        if not any(any_paths):
            raise ValueError(
                "At least one dataset path must be specified: "
                "`train_file_path`, `predict_file_path`, "
                "`train_table_path`, `predict_table_path`, "
                "`file_path`, or `table_path`"
            )
        return self

    def get_path(self, mode: t.Literal["train", "predict"]) -> t.Optional[str]:
        """Convenience accessor for the train/predict path."""
        if mode == "train":
            return self.train_file_path or self.train_table_path
        elif mode == "predict":
            return self.predict_file_path or self.predict_table_path
        else:
            raise ValueError(f"Unknown mode: {mode}")


class AllDatasetStagesConfig(pyd.BaseModel):
    bronze: dict[str, DatasetConfig]
    silver: dict[str, DatasetConfig]
    gold: dict[str, DatasetConfig]


class ModelConfig(pyd.BaseModel):
    experiment_id: str
    run_id: str
    framework: t.Optional[t.Literal["sklearn", "h2o"]] = "sklearn"

    @pyd.computed_field  # type: ignore[misc]
    @property
    def mlflow_model_uri(self) -> str:
        return f"runs:/{self.run_id}/model"


class PreprocessingConfig(pyd.BaseModel):
    selection: "SelectionConfig"
    checkpoint: "CheckpointConfig"
    target: "TargetConfig"
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


class CheckpointConfig(pyd.BaseModel):
    name: str = pyd.Field(default="checkpoint")
    params: dict[str, object] = pyd.Field(default_factory=dict)
    unit: t.Literal["credit", "year", "term", "semester"]
    value: int = pyd.Field(
        default=30,
        description=(
            "Number of checkpoint units (e.g. 1 year, 1 term/semester, 30 credits)"
        ),
    )
    optional_desc: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Optional description of the checkpoint beyond the unit and value. "
            "Used to provide further context for the particular institution and model. "
        ),
    )

    @pyd.field_validator("value")
    @classmethod
    def check_value_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Value must be greater than zero.")
        return v


class TargetConfig(pyd.BaseModel):
    name: str = pyd.Field(default="target")
    category: t.Literal["graduation", "retention"]
    params: dict[str, object] = pyd.Field(default_factory=dict)
    unit: t.Literal["credit", "year", "term", "semester", "pct_completion"]
    value: int = pyd.Field(
        default=120,
        description=(
            "Number of target units (e.g. 4 years, 4 terms, 120 credits, 150 completion %)"
        ),
    )
    optional_desc: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Optional description of the target beyond the unit and value. "
            "Used to provide further context for the particular institution and model. "
        ),
    )

    @pyd.field_validator("value")
    @classmethod
    def check_value_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Value must be greater than zero.")
        return v


class SelectionConfig(pyd.BaseModel):
    student_criteria: dict[str, object] = pyd.Field(
        default_factory=dict,
        description=(
            "Column name in modeling dataset mapped to one or more values that it must equal "
            "in order for the corresponding student to be considered 'eligible'. "
            "Multiple criteria are combined with a logical 'AND'."
        ),
    )
    student_criteria_aliases: dict[str, str] = pyd.Field(
        default_factory=dict,
        description="Human-readable display names for student_criteria keys",
    )

    @pyd.model_validator(mode="after")
    def validate_criteria_aliases(self) -> "SelectionConfig":
        criteria_keys = self.student_criteria.keys()
        alias_keys = self.student_criteria_aliases or {}

        missing = [k for k in criteria_keys if k not in alias_keys]
        if missing:
            raise ValueError(
                f"Missing display aliases in `student_criteria_aliases` for: {missing}"
            )
        return self


class ModelingConfig(pyd.BaseModel):
    feature_selection: t.Optional["FeatureSelectionConfig"] = None
    training: "TrainingConfig"
    evaluation: t.Optional["EvaluationConfig"] = None
    bias_mitigation: t.Optional["BiasMitigationConfig"] = None


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


class BiasMitigationConfig(pyd.BaseModel):
    student_group_col: str = pyd.Field(
        default="student_group",
        description="Column name in dataset to have a custom threshold set.",
    )
    student_group_col_alias: str = pyd.Field(
        default="Student Group",
        description="Human-readable display name for student_group column.",
    )
    student_group: str = pyd.Field(
        default="freshmen",
        description="Value in student_group column that has a custom threshold based on bias considerations.",
    )
    custom_threshold: float = pyd.Field(
        default=0.5,
        description="Threshold for student group based on bias considerations.",
    )


class InferenceConfig(pyd.BaseModel):
    num_top_features: int = pyd.Field(default=5)
    min_prob_pos_label: t.Optional[float] = 0.5
    background_data_sample: t.Optional[int] = 500
