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
    is_object_dtype,
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

    def __init__(self, add_missing_flags: bool = True):
        self.pipeline = None
        self.input_dtypes: t.Optional[dict[str, str]] = None
        self.add_missing_flags = add_missing_flags
        self.input_feature_names: t.Optional[list[str]] = None
        self.output_feature_names: t.Optional[list[str]] = None
        self.missing_flag_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> Pipeline:
        df = df.replace({None: np.nan})
        self.input_dtypes = {k: str(v) for k, v in df.dtypes.items()}
        self.input_feature_names = df.columns.tolist()

        if self.add_missing_flags:
            df = self._add_missingness_flags(df)
            self.missing_flag_cols = [
                c for c in df.columns if c.endswith("_missing_flag")
            ]
        else:
            self.missing_flag_cols = []

        pipeline = self._build_pipeline(df)
        pipeline.fit(df)
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.output_feature_names = self.pipeline.named_steps[
                "imputer"
            ].get_feature_names_out()

        return self.pipeline

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call `fit()` first.")

        df = df.replace({None: np.nan})

        # Filter/reorder to match original feature list
        if self.input_feature_names is not None:
            missing = set(self.input_feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required input features: {missing}")
            df = df[self.input_feature_names]

        # Add locked missingness flag columns
        if self.add_missing_flags:
            # Add any missing flag cols as False
            for col in self.missing_flag_cols:
                if col not in df:
                    df[col] = False
            df = self._add_missingness_flags(df)
            # Ensure all expected flags exist, in same order as fit-time
            for col in self.missing_flag_cols:
                if col not in df:
                    df[col] = False

        # Maintain column order from fit
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

        result = pd.DataFrame(
            transformed, columns=self.output_feature_names, index=df.index
        )

        # Restore data types
        for col in result.columns:
            try:
                result[col] = pd.to_numeric(result[col])
            except (ValueError, TypeError):
                pass

            if self.input_dtypes and col in self.input_dtypes:
                orig_dtype = self.input_dtypes[col]
                if is_bool_dtype(orig_dtype):
                    uniques = set(result[col].dropna().unique())
                    if uniques.issubset({0, 1, True, False}):
                        result[col] = result[col].astype(bool)

        self.validate(result)
        return result

    def _build_pipeline(self, df: pd.DataFrame) -> Pipeline:
        transformers = []
        skew_vals = df.select_dtypes(include="number").skew()

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                strategy = "passthrough"
            elif is_bool_dtype(df[col]):
                strategy = "most_frequent"
            elif is_numeric_dtype(df[col]):
                skew = skew_vals.get(col, 0)
                strategy = (
                    "median" if abs(skew) >= self.DEFAULT_SKEW_THRESHOLD else "mean"
                )
            elif is_categorical_dtype(df[col]) or is_object_dtype(df[col]):
                strategy = "most_frequent"
            else:
                strategy = "most_frequent"

            if strategy == "passthrough":
                transformers.append((col, "passthrough", [col]))
            else:
                imputer = SimpleImputer(strategy=strategy)
                transformers.append((col, imputer, [col]))

        ct = ColumnTransformer(
            transformers, remainder="passthrough", verbose_feature_names_out=False
        )
        return Pipeline([("imputer", ct)])

    def log_pipeline(self, artifact_path: str) -> None:
        """
        Logs the fitted pipeline and input dtypes to MLflow as artifacts.

        Args:
            artifact_path: MLflow artifact subdirectory (e.g., "sklearn_imputer")
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save pipeline
            pipeline_path = os.path.join(tmpdir, self.PIPELINE_FILENAME)
            joblib.dump(self.pipeline, pipeline_path)

            # Save input dtypes if available
            if self.input_dtypes is not None:
                dtypes_path = os.path.join(tmpdir, "input_dtypes.json")
                with open(dtypes_path, "w") as f:
                    json.dump(self.input_dtypes, f, indent=2)
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
            if self.add_missing_flags:
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

    def validate(self, df: pd.DataFrame, raise_error: bool = True) -> bool:
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].index.tolist()

        if missing_cols:
            msg = f"Transformed data still contains nulls in: {missing_cols}"
            if raise_error:
                raise ValueError(msg)
            LOGGER.warning(msg)
            return False

        return True

    def _add_missingness_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().any():
                df[f"{col}_missing_flag"] = df[col].isnull()
        return df

    @classmethod
    def load(
        cls, run_id: str, artifact_path: str = "sklearn_imputer"
    ) -> "SklearnImputerWrapper":
        """
        Load a trained SklearnImputerWrapper from MLflow, including pipeline and input metadata.
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
                instance.input_dtypes = json.load(f)
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
        raise_on_nulls: bool = True,
    ) -> pd.DataFrame:
        """
        Load a trained SklearnImputerWrapper from MLflow and apply it to the given DataFrame.
        """
        instance = cls.load(run_id=run_id, artifact_path=artifact_path)

        # Filter and/or reorder columns if input_feature_names are available
        if instance.input_feature_names:
            missing = set(instance.input_feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required input features: {missing}")

            df = df[instance.input_feature_names]

        transformed = instance.transform(df)
        instance.validate(transformed, raise_error=raise_on_nulls)
        return transformed
