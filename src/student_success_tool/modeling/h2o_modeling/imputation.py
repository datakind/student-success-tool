import pandas as pd
import numpy as np
import json

import typing as t
import os
import tempfile
import logging
import mlflow
import joblib
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_string_dtype,
    is_integer_dtype,
    is_object_dtype,
    pandas_dtype,
)

LOGGER = logging.getLogger(__name__)


class SklearnImputerWrapper:
    """
    A leakage-safe imputer using sklearn's SimpleImputer and ColumnTransformer,
    with skew-aware numeric strategy assignment, optional missingness flags,
    and MLflow-based artifact logging.
    """

    DEFAULT_SKEW_THRESHOLD = 0.5
    PIPELINE_FILENAME = "imputer_pipeline.joblib"

    def __init__(self):
        self.pipeline = None
        self.input_dtypes: t.Optional[dict[str, t.Any]] = None
        self.input_feature_names: t.Optional[list[str]] = None
        self.output_feature_names: t.Optional[list[str]] = None
        self.missing_flag_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> Pipeline:
        """
        Fit the imputer pipeline on a DataFrame. Replaces `None` with NaN, records dtypes and feature names,
        optionally adds missingness flags, and builds a column-wise imputer
        with appropriate strategies for each dtype.

        Parameters:
            df: Input DataFrame to fit on.

        Returns:
            The fitted scikit-learn `Pipeline` instance.
        """
        # record originals before coercion so we can cast back in transform()
        self.input_dtypes = df.dtypes.to_dict()
        self.input_feature_names = df.columns.tolist()

        # normalize & coerce for sklearn
        df = self._normalize_missing(df)
        df = self._coerce_extension_types_for_sklearn(df)

        # add flags after normalization/coercion (flags have no NA)
        df = self._add_missingness_flags(df)
        self.missing_flag_cols = [c for c in df.columns if c.endswith("_missing_flag")]

        pipeline = self._build_pipeline(df)
        pipeline.fit(df)
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.output_feature_names = self.pipeline.named_steps[
                "imputer"
            ].get_feature_names_out()

        return self.pipeline

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted imputer pipeline to new data.
        Ensures column alignment with training-time features, adds missingness
        flags if enabled, maintains original row order, and restores data types
        after imputation. Raises 'ValueError' if  the pipeline has not been fitted, or if required
        features are missing, or if row counts mismatch.

        Parameters:
            df: DataFrame to transform.

        Returns:
            Imputed DataFrame with same index as input.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call `fit()` first.")

        assert self.input_dtypes is not None, (
            "input_dtypes missing; call fit() before transform()."
        )

        orig_index = df.index  # Lock in row order
        df_original = df.copy()

        df = self._normalize_missing(df)
        df = self._coerce_extension_types_for_sklearn(df)

        # Compute extra columns (e.g. student_id_col) before subsetting so we can reattach later
        raw_features = list(self.input_feature_names or [])
        missing = set(raw_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required input features: {missing}")
        extra_cols = [c for c in df_original.columns if c not in raw_features]

        # Filter/reorder to match training-time input
        if self.input_feature_names is not None:
            missing = set(self.input_feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required input features: {missing}")
            df = df[self.input_feature_names]

        # Add missingness flags
        df = self._add_missingness_flags(df)

        # Ensure missingness flags from fit-time exist
        for col in self.missing_flag_cols:
            if col not in df:
                df[col] = False

        # Maintain exact column order from fit
        if self.input_feature_names:
            df = df[self.input_feature_names + self.missing_flag_cols]

        transformed = self.pipeline.transform(df)

        # Row count safety check
        if transformed.shape[0] != df.shape[0]:
            raise ValueError(
                f"Row count mismatch after imputation: input had {df.shape[0]} rows, "
                f"output has {transformed.shape[0]} rows"
            )

        if self.output_feature_names is None:
            raise ValueError(
                "Output feature names not set. Did you forget to call `fit()`?"
            )

        # Build DataFrame with original index
        result = pd.DataFrame(
            transformed, columns=self.output_feature_names, index=orig_index
        )

        for col, orig_dtype in self.input_dtypes.items():
            if col not in result.columns:
                continue

            if is_bool_dtype(orig_dtype):
                # Imputer may output floats 0/1 or bools, coerce safely to pandas boolean dtype
                s = pd.Series(result[col])
                # handle strings "0"/"1" or "True"/"False" and numerics
                s = pd.to_numeric(s, errors="coerce").round()
                result[col] = (
                    s.astype("Int64").map({0: False, 1: True}).astype("boolean")
                )

            elif is_numeric_dtype(orig_dtype):
                # Keep numeric columns numeric
                result[col] = pd.to_numeric(result[col], errors="coerce")

            else:
                # Originally non-numeric -> keep as pandas string dtype (prevents "1010" -> 1010)
                result[col] = pd.Series(result[col]).astype("string")

        # Ensure missing_flag columns are boolean
        for col in self.missing_flag_cols:
            if col in result.columns:
                result[col] = pd.Series(result[col]).astype("boolean")

        # Reattach extras (avoid name collisions with imputed output)
        extra_cols = [c for c in extra_cols if c not in result.columns]
        if extra_cols:
            result = pd.concat(
                [result, df_original.loc[orig_index, extra_cols]], axis=1
            )

        # Validate only the imputed columns
        self.validate(result[self.output_feature_names])
        return result

    def _build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        """
        Construct an imputation pipeline with dtype and skew-aware strategies.

        For each column in the DataFrame, an imputation strategy is chosen based on:
        - No missing values-> passthrough (no transformation)
        - Boolean dtype -> most frequent value
        - Numeric dtype -> median if absolute skewness ≥ DEFAULT_SKEW_THRESHOLD, else mean
        - Categorical or object dtype -> most frequent value
        - Other types -> most frequent value

        Parameters:
            df: Input DataFrame used to determine dtypes, skewness, and missingness patterns.

        Returns:
            An unfitted scikit-learn `Pipeline` containing a `ColumnTransformer`
            with per-column `SimpleImputer` instances or passthroughs.
        """
        transformers = []
        skew_vals = df.select_dtypes(include="number").skew(numeric_only=True)

        for col in df.columns:
            s = df[col]
            n_obs = s.notna().sum()

            # 1) All NaNs -> constant fill
            if n_obs == 0:
                if is_bool_dtype(s.dtype):
                    imputer = SimpleImputer(strategy="constant", fill_value=False)
                elif is_numeric_dtype(s):
                    fill_val = 0 if is_integer_dtype(s) else 0.0
                    imputer = SimpleImputer(strategy="constant", fill_value=fill_val)
                else:
                    imputer = SimpleImputer(strategy="constant", fill_value="missing")
                transformers.append((col, imputer, [col]))
                continue

            # 2) No NaNs -> passthrough
            if not s.isna().any():
                transformers.append((col, "passthrough", [col]))
                continue

            # 3) Some NaNs -> choose strategy by dtype (check skew for numerics)
            if is_bool_dtype(s.dtype):
                strategy = "most_frequent"
            elif is_numeric_dtype(s):
                skew = skew_vals.get(col, 0)
                strategy = (
                    "median" if abs(skew) >= self.DEFAULT_SKEW_THRESHOLD else "mean"
                )
            elif is_categorical_dtype(s) or is_object_dtype(s):
                strategy = "most_frequent"
            else:
                strategy = "most_frequent"

            imputer = SimpleImputer(strategy=strategy)
            transformers.append((col, imputer, [col]))

        ct = ColumnTransformer(
            transformers, remainder="passthrough", verbose_feature_names_out=False
        )
        return Pipeline([("imputer", ct)])

    def log_pipeline(self, artifact_path: str) -> None:
        """
        Logs the fitted pipeline and input dtypes to MLflow as artifacts.

        Parameters:
            artifact_path: MLflow artifact subdirectory (e.g., "sklearn_imputer")
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline not fitted. Call `fit()` before `log_pipeline()`."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save pipeline
            pipeline_path = os.path.join(tmpdir, self.PIPELINE_FILENAME)
            joblib.dump(self.pipeline, pipeline_path)

            # Save input dtypes if available
            if self.input_dtypes is not None:
                dtypes_path = os.path.join(tmpdir, "input_dtypes.json")
                with open(dtypes_path, "w") as f:
                    json.dump(
                        {k: str(v) for k, v in self.input_dtypes.items()}, f, indent=2
                    )
            else:
                dtypes_path = None

            # Save input feature names if available
            if self.input_feature_names is not None:
                features_path = os.path.join(tmpdir, "input_feature_names.json")
                with open(features_path, "w") as f:
                    json.dump(self.input_feature_names, f, indent=2)
            else:
                features_path = None

            # Save missing_flag_cols if available
            if self.missing_flag_cols is not None:
                flags_path = os.path.join(tmpdir, "missing_flag_cols.json")
                with open(flags_path, "w") as f:
                    json.dump(self.missing_flag_cols, f, indent=2)
            else:
                flags_path = None

            def log_artifacts():
                mlflow.log_artifact(pipeline_path, artifact_path=artifact_path)
                LOGGER.debug(
                    f"Logged pipeline to MLflow at: {artifact_path}/{self.PIPELINE_FILENAME}"
                )
                if dtypes_path:
                    mlflow.log_artifact(dtypes_path, artifact_path=artifact_path)
                    LOGGER.debug(
                        f"Logged input_dtypes to MLflow at: {artifact_path}/input_dtypes.json"
                    )
                if features_path:
                    mlflow.log_artifact(features_path, artifact_path=artifact_path)
                    LOGGER.debug(
                        f"Logged input_feature_names to MLflow at: {artifact_path}/input_feature_names.json"
                    )
                if flags_path:
                    mlflow.log_artifact(flags_path, artifact_path=artifact_path)
                    LOGGER.debug(
                        f"Logged missing_flag_cols to MLflow at: {artifact_path}/missing_flag_cols.json"
                    )

            # Respect existing run context or start a new one
            if mlflow.active_run():
                log_artifacts()
            else:
                with mlflow.start_run(run_name="sklearn_preprocessing"):
                    log_artifacts()

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Check that the DataFrame contains no null values after imputation.

        Parameters:
            df: DataFrame to validate.

        Returns:
            True if no nulls found, False otherwise.
        """
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].index.tolist()

        if missing_cols:
            raise ValueError(
                f"Transformed data still contains nulls in: {missing_cols}"
            )
        return True

    def _add_missingness_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().any():
                df[f"{col}_missing_flag"] = df[col].isnull().astype(bool)
        return df

    @classmethod
    def load(
        cls, run_id: str, artifact_path: str = "sklearn_imputer"
    ) -> "SklearnImputerWrapper":
        """
        Load a trained imputer pipeline from MLflow.
        Restores the fitted pipeline and associated metadata, including input
        dtypes, feature names, and missingness flag columns.

        Parameters:
            run_id: The MLflow run ID from which to load artifacts.
            artifact_path: The artifact subdirectory used when logging.

        Returns:
            An instance of `SklearnImputerWrapper` with loaded state.
        """
        instance = cls()
        LOGGER.info(f"Loading pipeline from MLflow run {run_id}...")

        # Load pipeline
        local_pipeline_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/{cls.PIPELINE_FILENAME}"
        )
        instance.pipeline = joblib.load(local_pipeline_path)

        # Load input_dtypes
        try:
            dtypes_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{artifact_path}/input_dtypes.json"
            )
            with open(dtypes_path) as f:
                loaded = json.load(f)
            # convert back to dtype objects
            instance.input_dtypes = {k: pandas_dtype(v) for k, v in loaded.items()}
            LOGGER.info("Successfully loaded input_dtypes from MLflow.")
        except Exception as e:
            LOGGER.warning(f"Could not load input_dtypes.json for run {run_id}. ({e})")
            instance.input_dtypes = None

        # Load input_feature_names
        try:
            features_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{artifact_path}/input_feature_names.json"
            )
            with open(features_path) as f:
                instance.input_feature_names = json.load(f)
            LOGGER.info("Successfully loaded input_feature_names from MLflow.")
        except Exception as e:
            LOGGER.warning(
                f"Could not load input_feature_names.json for run {run_id}. "
                f"Transformation input alignment may be incorrect. ({e})"
            )
            instance.input_feature_names = None

        # Load missingness_flags from fit()
        try:
            flags_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{artifact_path}/missing_flag_cols.json"
            )
            with open(flags_path) as f:
                instance.missing_flag_cols = json.load(f)
            LOGGER.info("Successfully loaded missing_flag_cols from MLflow.")
        except Exception as e:
            LOGGER.warning(
                f"Could not load missing_flag_cols.json for run {run_id}. "
                f"Missing flags may have been generated incorrectly. ({e})"
            )
            instance.missing_flag_cols = []

        # Restore output feature names if possible
        pipeline = instance.pipeline
        if pipeline is not None:
            try:
                instance.output_feature_names = pipeline.named_steps[
                    "imputer"
                ].get_feature_names_out()
            except Exception:
                instance.output_feature_names = None
        else:
            instance.output_feature_names = None

        return instance

    @classmethod
    def load_and_transform(
        cls,
        df: pd.DataFrame,
        *,
        run_id: str,
        artifact_path: str = "sklearn_imputer",
    ) -> pd.DataFrame:
        """
        Load a trained imputer from MLflow and apply it to data.

        Parameters:
            df: DataFrame to transform.
            run_id: MLflow run ID for loading the pipeline.
            artifact_path: Artifact subdirectory used when logging.

        Returns:
            Imputed DataFrame with same index as input.
        """
        instance = cls.load(run_id=run_id, artifact_path=artifact_path)
        transformed = instance.transform(df)
        if instance.output_feature_names is not None:
            instance.validate(transformed[instance.output_feature_names])
        return transformed

    def _coerce_extension_types_for_sklearn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make all columns sklearn-safe (no pd.NA makes it into object arrays).
        - boolean/BooleanDtype  -> float64 (1.0/0.0/np.nan)
        - nullable integers      -> float64 with np.nan
        - pandas string dtype    -> object with np.nan
        - pandas Float64Dtype    -> float64 (plain numpy)
        - categories             -> object with np.nan
        """
        df = df.copy()
        for col in df.columns:
            dt = df[col].dtype

            # 1) Booleans -> float64
            if is_bool_dtype(dt):
                df[col] = df[col].astype("float64")
                continue

            # 2) Nullable ints -> float64
            if is_integer_dtype(dt) and str(dt).startswith(("Int", "UInt")):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
                continue

            # 3) Pandas Float64Dtype -> float64 (plain)
            if str(dt) == "Float64":
                df[col] = df[col].astype("float64")
                continue

            # 4) Pandas StringDtype -> object (ensure np.nan, not <NA>)
            if is_string_dtype(dt) or str(dt).startswith("string"):
                s = df[col].astype("object")
                df[col] = s.where(~pd.isna(s), np.nan)
                continue

            # 5) Categoricals -> object (ensure np.nan)
            if is_categorical_dtype(dt):
                s = df[col].astype("object")
                df[col] = s.where(~pd.isna(s), np.nan)
                continue

            # Others (float64, float32, plain object already w/ np.nan) are fine.
        return df

    def _normalize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert any logical missing to np.nan without comparing to pd.NA
        df = df.replace({None: np.nan})
        df = df.mask(df.isna(), np.nan)
        return df
