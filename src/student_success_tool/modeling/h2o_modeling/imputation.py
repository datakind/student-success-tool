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
        self.original_dtypes: t.Optional[dict[str, str]] = None
        self.add_missing_flags = add_missing_flags
        self.feature_names = None

    def fit(self, df: pd.DataFrame, artifact_path: str = "sklearn_imputer") -> Pipeline:
        df = df.replace({None: np.nan})
        self.original_dtypes = {k: str(v) for k, v in df.dtypes.items()}

        if self.add_missing_flags:
            df = self._add_missingness_flags(df)

        pipeline = self._build_pipeline(df)
        pipeline.fit(df)
        self.pipeline = pipeline
        if self.pipeline is not None:
            self.feature_names = self.pipeline.named_steps[
                "imputer"
            ].get_feature_names_out()

        return self.pipeline

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call `fit()` first.")

        df = df.replace({None: np.nan})

        if self.add_missing_flags:
            df = self._add_missingness_flags(df)

        transformed = self.pipeline.transform(df)
        result = pd.DataFrame(transformed, columns=self.feature_names, index=df.index)

        # Restore dtypes
        for col in result.columns:
            try:
                result[col] = pd.to_numeric(result[col])
            except (ValueError, TypeError):
                pass

            if self.original_dtypes and col in self.original_dtypes:
                orig_dtype = self.original_dtypes[col]
                if is_bool_dtype(orig_dtype):
                    uniques = set(result[col].dropna().unique())
                    if uniques.issubset({0, 1, True, False}):
                        result[col] = result[col].astype(bool)

            if result[col].dtype == "object":
                sample_vals = result[col].dropna().astype(str).head(10)
                if all(v.replace(".", "", 1).isdigit() for v in sample_vals):
                    LOGGER.warning(
                        f"Column '{col}' is object but contains numeric-looking values after imputation."
                    )

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
        Logs the fitted pipeline and original dtypes to MLflow as artifacts.

        Args:
            artifact_path: MLflow artifact subdirectory (e.g., "sklearn_imputer")
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save pipeline
            pipeline_path = os.path.join(tmpdir, self.PIPELINE_FILENAME)
            joblib.dump(self.pipeline, pipeline_path)

            # Save original dtypes if available
            if self.original_dtypes is not None:
                dtypes_path = os.path.join(tmpdir, "original_dtypes.json")
                with open(dtypes_path, "w") as f:
                    json.dump(self.original_dtypes, f, indent=2)
            else:
                dtypes_path = None

            def log_artifacts():
                mlflow.log_artifact(pipeline_path, artifact_path=artifact_path)
                LOGGER.debug(
                    f"Logged pipeline to MLflow at: {artifact_path}/{self.PIPELINE_FILENAME}"
                )
                if dtypes_path:
                    mlflow.log_artifact(dtypes_path, artifact_path=artifact_path)
                    LOGGER.debug(
                        f"Logged original_dtypes to MLflow at: {artifact_path}/original_dtypes.json"
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
        Load a trained SklearnImputerWrapper from MLflow, including pipeline and original dtypes.

        Args:
            run_id: MLflow run ID from training time.
            artifact_path: Artifact subdirectory where the imputer and metadata were logged.

        Returns:
            SklearnImputerWrapper instance with pipeline and dtypes restored.
        """
        instance = cls()
        LOGGER.info(f"Loading pipeline from MLflow run {run_id}...")

        # Download pipeline artifact
        local_pipeline_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/{cls.PIPELINE_FILENAME}"
        )
        instance.pipeline = joblib.load(local_pipeline_path)

        # Download original_dtypes JSON
        try:
            dtypes_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"{artifact_path}/original_dtypes.json"
            )
            with open(dtypes_path) as f:
                instance.original_dtypes = json.load(f)
            LOGGER.info("Successfully loaded original_dtypes from MLflow.")
        except Exception as e:
            LOGGER.warning(
                f"Could not load original_dtypes.json for run {run_id}. Inference may have reduced type fidelity. ({e})"
            )
            instance.original_dtypes = None

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

        Args:
            df: Raw input DataFrame to transform.
            run_id: MLflow run ID where the imputer was logged.
            artifact_path: Path within the run where pipeline and metadata are stored.
            raise_on_nulls: If True, raises an error if transformed data has missing values.

        Returns:
            Transformed DataFrame with imputed and dtype-aligned features.
        """
        instance = cls.load(run_id=run_id, artifact_path=artifact_path)
        transformed = instance.transform(df)
        instance.validate(transformed, raise_error=raise_on_nulls)
        return transformed
