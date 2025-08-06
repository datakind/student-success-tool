import pandas as pd
import numpy as np
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
    with skew-aware numeric strategy assignment and MLflow-based artifact logging.

    Automatically saves the fitted pipeline with joblib.
    """

    DEFAULT_SKEW_THRESHOLD = 0.5
    PIPELINE_FILENAME = "imputer_pipeline.joblib"

    def __init__(self):
        self.pipeline = None

    def fit(self, df: pd.DataFrame, artifact_path: str = "sklearn_imputer") -> Pipeline:
        """
        Assigns imputation strategies, builds a pipeline, fits it, and logs via MLflow.

        Args:
            df: training DataFrame
            artifact_path: MLflow artifact path to save pipeline

        Returns:
            Fitted sklearn Pipeline
        """
        df = df.replace({None: np.nan})
        pipeline = self._build_pipeline(df)
        pipeline.fit(df)
        self.pipeline = pipeline

        LOGGER.info("Saving pipeline to MLflow...")
        self.log_pipeline(artifact_path)
        return self.pipeline

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call `fit()` first.")

        # Replace None with np.nan so SimpleImputer can handle it
        df = df.replace({None: np.nan})

        # Run pipeline (ColumnTransformer + SimpleImputer)
        transformed = self.pipeline.transform(df)

        # Rebuild DataFrame with original column names
        result = pd.DataFrame(transformed, columns=df.columns, index=df.index)

        # Try to coerce each column back to numeric if appropriate
        for col in result.columns:
            # Try numeric conversion (e.g., float/int)
            try:
                result[col] = pd.to_numeric(result[col])
            except (ValueError, TypeError):
                pass

            # Try bool conversion for columns that were originally boolean
            if is_bool_dtype(df[col]):
                uniques = set(result[col].dropna().unique())
                if uniques.issubset({0, 1, True, False}):
                    result[col] = result[col].astype(bool)

            #  Log warning if still object but looks numeric
            if result[col].dtype == "object":
                sample_vals = result[col].dropna().astype(str).head(10)
                if all(v.replace(".", "", 1).isdigit() for v in sample_vals):
                    LOGGER.warning(
                        f"Column '{col}' is object but contains numeric-looking values after imputation."
                    )

        self.validate(result)  # Check for leftover nulls
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

    def log_pipeline(self, artifact_path: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, self.PIPELINE_FILENAME)
            joblib.dump(self.pipeline, file_path)

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name="sklearn_preprocessing"):
                mlflow.log_artifact(file_path, artifact_path=artifact_path)

            LOGGER.info(
                f"Logged pipeline to MLflow: {artifact_path}/{self.PIPELINE_FILENAME}"
            )

    def validate(self, df: pd.DataFrame, raise_error: bool = True) -> bool:
        """Checks if any missing values remain after transform."""
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].index.tolist()

        if missing_cols:
            msg = f"Transformed data still contains nulls in: {missing_cols}"
            if raise_error:
                raise ValueError(msg)
            LOGGER.warning(msg)
            return False

        return True

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
