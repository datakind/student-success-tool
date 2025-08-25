import typing as t
from mlflow.tracking import MlflowClient

# internal SST modules
from ...modeling import h2o_modeling

from ...configs.h2o_configs.custom import CustomProjectConfig
from .base import ModelCard
from ..sections.custom import register_sections as register_custom_sections
from ..utils import utils


class H2OCustomModelCard(ModelCard[CustomProjectConfig]):
    def __init__(
        self,
        config: CustomProjectConfig,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes custom model card by enforcing a custom project config.
        Otherwise, this class inherits and is functionally the same as the
        base ModelCard class.
        """
        if not isinstance(config, CustomProjectConfig):  # type guard
            raise TypeError("Expected config to be of type CustomProjectConfig")

        super().__init__(config, catalog, model_name, assets_path, mlflow_client)

    def _register_sections(self):
        """
        Register cusom-specific sections.
        """
        # Clearing registry for overrides
        self.section_registry.clear()

        # Register custom-specific sections
        register_custom_sections(self, self.section_registry)

    def load_model(self):
        """
        Loads the MLflow model from the MLflow client based on the MLflow model URI.
        Also assigns the run ID and experiment ID from the config.
        """
        model_cfg = self.cfg.model
        if not model_cfg:
            raise ValueError(f"Model configuration for '{self.model_name}' is missing.")
        if not all(
            [model_cfg.mlflow_model_uri, model_cfg.run_id, model_cfg.experiment_id]
        ):
            raise ValueError(
                f"Incomplete model config for '{self.model_name}': "
                f"URI, run_id, or experiment_id missing."
            )

        self.model = h2o_modeling.utils.load_h2o_model(model_cfg.run_id)
        self.run_id = model_cfg.run_id
        self.experiment_id = model_cfg.experiment_id

    def extract_training_data(self):
        """
        Extracts the training data from the MLflow run utilizing SST internal subpackages (modeling).
        """
        self.modeling_data = h2o_modeling.evaluation.extract_training_data_from_model(
            self.experiment_id
        )
        self.training_data = self.modeling_data
        if self.cfg.split_col:
            if self.cfg.split_col not in self.modeling_data.columns:
                raise ValueError(
                    f"Configured split_col '{self.cfg.split_col}' is not present in modeling data columns: "
                    f"{list(self.modeling_data.columns)}"
                )
            self.training_data = self.modeling_data[
                self.modeling_data[self.cfg.split_col] == "train"
            ]
        self.context["training_dataset_size"] = self.training_data.shape[0]
        self.context["num_runs_in_experiment"] = (
            h2o_modeling.evaluation.extract_number_of_runs_from_model_training(
                self.experiment_id
            )
        )

    def get_model_plots(self) -> dict[str, str]:
        """
        Collects model plots from the MLflow run, downloads them locally. These will later be
        rendered in the template.

        Returns:
            A dictionary with the keys as the plot names called in the template
            and the values are inline HTML (since these are all images) for each
            of the artifacts.
        """
        plots = {
            "model_comparison_plot": (
                "Model Comparison",
                "model_comparison.png",
                "125mm",
            ),
            "test_calibration_curve": (
                "Test Calibration Curve",
                "calibration/test_calibration.png",
                "125mm",
            ),
            "test_roc_curve": (
                "Test ROC Curve",
                "test_roc_curve_plot.png",
                "125mm",
            ),
            "test_confusion_matrix": (
                "Test Confusion Matrix",
                "test_confusion_matrix.png",
                "125mm",
            ),
            "test_histogram": (
                "Test Histogram",
                "preds/test_hist.png",
                "125mm",
            ),
            "feature_importances_by_shap_plot": (
                "Feature Importances",
                "h2o_feature_importances_by_shap_plot.png",
                "150mm",
            ),
        }
        return {
            key: utils.download_artifact(
                run_id=self.run_id,
                description=description,
                artifact_path=path,
                local_folder=self.assets_folder,
                fixed_width=width,
            )
            or ""
            for key, (description, path, width) in plots.items()
        }
