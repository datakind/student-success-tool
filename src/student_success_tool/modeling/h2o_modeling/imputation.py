import pandas as pd
import numpy as np

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
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, self.PIPELINE_FILENAME)
            joblib.dump(self.pipeline, file_path)

            if mlflow.active_run():
                # Use the current run (do not start a new one)
                mlflow.log_artifact(file_path, artifact_path=artifact_path)
            else:
                # Only start a new run if one isn't active
                with mlflow.start_run("sklearn_preprocessing"):
                    mlflow.log_artifact(file_path, artifact_path=artifact_path)

            LOGGER.debug(
                f"Logged pipeline to MLflow: {artifact_path}/{self.PIPELINE_FILENAME}"
            )

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
        instance = cls()
        LOGGER.info(f"Loading pipeline from MLflow run {run_id}...")

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/{cls.PIPELINE_FILENAME}"
        )
        instance.pipeline = joblib.load(local_path)
        return instance
