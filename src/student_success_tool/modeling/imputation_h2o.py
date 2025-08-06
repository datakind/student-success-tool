import h2o
import json
import tempfile
import os
import logging
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_object_dtype,
)
import mlflow

LOGGER = logging.getLogger(__name__)


class H2OImputerWrapper:
    """
    A wrapper for column-wise imputation using
    with skew-aware strategy assignment and MLflow logging.

    Supports:
    - mean/median for numeric columns (based on skewness)
    - mode for booleans and categoricals
    """

    DEFAULT_SKEW_THRESHOLD = 0.5
    IMPUTATION_MAP_FILENAME = "imputation_map.json"

    def __init__(self):
        self.strategy_map = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        train_h2o: h2o.H2OFrame,
        valid_h2o: h2o.H2OFrame,
        test_h2o: h2o.H2OFrame,
        artifact_path: str = "h2o_imputer",
    ) -> tuple[h2o.H2OFrame, h2o.H2OFrame, h2o.H2OFrame]:
        """
        Assign imputation strategies, apply imputation to all splits, and log strategy map.

        Args:
            train_df: Pandas DataFrame (used for dtype/skewness).
            train_h2o: H2OFrame of training data.
            valid_h2o: H2OFrame of validation data.
            test_h2o: H2OFrame of test data.
            artifact_path: MLflow artifact directory.

        Returns:
            Tuple of imputed H2OFrames: (train, valid, test)
        """
        LOGGER.info("Assigning imputation strategies based on skew and dtype...")
        self.strategy_map = self._assign_strategies(train_df)

        LOGGER.info("Applying imputations to all H2OFrame splits...")
        train_h2o = self._apply_imputation(train_h2o)
        valid_h2o = self._apply_imputation(valid_h2o)
        test_h2o = self._apply_imputation(test_h2o)

        # Ensure no missing values remain
        self._assert_no_missing(train_h2o, name="train_h2o")
        self._assert_no_missing(valid_h2o, name="valid_h2o")
        self._assert_no_missing(test_h2o, name="test_h2o")


        LOGGER.info("Logging strategy map to MLflow...")
        self.log(artifact_path)

        return train_h2o, valid_h2o, test_h2o

    def transform(self, h2o_frame: h2o.H2OFrame) -> h2o.H2OFrame:
        """
        Apply stored imputation strategies to a new H2OFrame.

        Raises:
            ValueError if strategy_map is not set.

        Returns:
            Imputed H2OFrame.
        """
        if not self.strategy_map:
            raise ValueError(
                "No strategy_map found. Call fit() or load() before transform()."
            )
        return self._apply_imputation(h2o_frame)

    def log(self, artifact_path: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = os.path.join(tmpdir, self.IMPUTATION_MAP_FILENAME)
            with open(map_path, "w") as f:
                json.dump(self.strategy_map, f, indent=2)

            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name="h2o_preprocessing"):
                mlflow.log_artifact(map_path, artifact_path=artifact_path)

            LOGGER.info(
                f"Logged strategy map to MLflow: {artifact_path}/{self.IMPUTATION_MAP_FILENAME}"
            )

    def _apply_imputation(self, h2o_frame: h2o.H2OFrame) -> h2o.H2OFrame:
        for col, config in self.strategy_map.items():
            strategy = config["strategy"]
            value = config["value"]

            try:
                assert isinstance(value, (int, float, str, bool)), (
                    f"{col} has non-scalar value: {value} (type: {type(value)})"
                )

                if h2o_frame[col].isfactor()[0]:
                    levels_list = h2o_frame[col].levels()
                    if levels_list and value not in levels_list[0]:
                        LOGGER.info(
                            f"Converting '{col}' to character for imputation "
                            f"(value '{value}' not in levels {levels_list[0]})"
                        )
                        h2o_frame[col] = h2o_frame[col].ascharacter()

                # Refetch AST after any potential conversion
                if h2o_frame[col].isna().sum() == 0:
                    continue

                # # Coerce value to string if column is string
                # col_type = h2o_frame.type(col)
                # if col_type == "string" and not isinstance(value, str):
                #     LOGGER.info(f"Coercing value for column '{col}' to string: {value}")
                #     value = str(value)

                isna_col = h2o_frame[col].isna()
                h2o_frame[col] = isna_col.ifelse(value, h2o_frame[col])

            except Exception as e:
                LOGGER.warning(
                    f"Failed to impute '{col}' with '{strategy}' and '{value}'  (type: {type(value)}): {e}"
                )
        return h2o_frame


    def _assign_strategies(self, df: pd.DataFrame) -> dict:
        skew_vals = df.select_dtypes(include="number").skew()
        strategy_map = {}

        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue

            if is_bool_dtype(df[col]):
                strategy = "mode"
                value = df[col].mode(dropna=True).iloc[0]
            elif is_numeric_dtype(df[col]):
                skew = skew_vals.get(col, 0)
                strategy = (
                    "median" if abs(skew) >= self.DEFAULT_SKEW_THRESHOLD else "mean"
                )
                value = float(
                    df[col].median() if strategy == "median" else df[col].mean()
                )
            elif is_categorical_dtype(df[col]) or is_object_dtype(df[col]):
                strategy = "mode"
                value = df[col].mode(dropna=True).iloc[0]
            else:
                strategy = "mode"
                value = df[col].mode(dropna=True).iloc[0]

            strategy_map[col] = {"strategy": strategy, "value": value}

        LOGGER.debug(f"Assigned strategy map: {strategy_map}")
        return strategy_map

    @classmethod
    def load(
        cls, run_id: str, artifact_path: str = "h2o_imputer"
    ) -> "H2OImputerWrapper":
        """
        Load imputation strategy map from MLflow artifacts.

        Args:
            run_id: MLflow run ID.
            artifact_path: Path to imputation_map.json.

        Returns:
            Initialized H2OImputerWrapper with loaded strategy map.
        """
        instance = cls()
        LOGGER.info(f"Loading strategy map from MLflow run {run_id}...")
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/imputation_map.json"
        )
        with open(local_path) as f:
            instance.strategy_map = json.load(f)
        return instance

    def _assert_no_missing(self, h2o_frame, name: str = "frame"):
        missing_cols = [col for col in h2o_frame.columns if h2o_frame[col].isna().sum() > 0]
        if missing_cols:
            raise ValueError(f"{name} still has missing values in: {missing_cols}")