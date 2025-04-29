import os
import mlflow
import logging
from mlflow.tracking import MlflowClient
from datetime import datetime
import importlib.abc
from importlib.resources import files

# internal SST modules
from .. import dataio, modeling

# relative imports in 'reporting' module 
from .sections import register_sections
from .sections.registry import SectionRegistry
from .utils import utils
from .utils.formatting import Formatting

LOGGER = logging.getLogger(__name__)

class ModelCard:
    DEFAULT_ASSETS_FOLDER = "card_assets"
    TEMPLATE_FILENAME = "model-card-TEMPLATE.md"
    LOGO_FILENAME = "logo.png"

    def __init__(self, config, uc_model_name, assets_path=None):
        self.cfg = config
        self.uc_model_name = uc_model_name
        self.model_name = self._extract_model_name(uc_model_name)

        self.client = MlflowClient()
        self.section_registry = SectionRegistry()
        self.format = Formatting()
        self.context = {}

        self.assets_folder = assets_path or self.DEFAULT_ASSETS_FOLDER
        self.output_path = self._build_output_path()
        self.template_path = self._resolve_template(self.TEMPLATE_FILENAME)
        self.logo_path = self._resolve_asset(self.LOGO_FILENAME)

    def build(self):
        self.load_model()
        self.find_model_version()
        self.extract_training_data()
        self._register_sections()
        self.collect_metadata()
        self.render()

    def load_model(self):
        model_cfg = self.cfg.models[self.model_name]
        self.model = dataio.models.load_mlflow_model(
            model_cfg.mlflow_model_uri,
            model_cfg.framework,
        )
        self.run_id = model_cfg.run_id
        self.experiment_id = model_cfg.experiment_id

    def find_model_version(self):
        versions = self.client.search_model_versions(f"name='{self.uc_model_name}'")
        for v in versions:
            if v.run_id == self.run_id:
                self.context["version_number"] = v.version
                return
        raise ValueError(f"No registered model version found for run_id={self.run_id}")

    def extract_training_data(self):
        self.modeling_data = modeling.evaluation.extract_training_data_from_model(self.experiment_id)
        self.training_data = self.modeling_data
        if self.cfg.split_col:
            self.training_data = self.modeling_data[
                self.modeling_data[self.cfg.split_col] == "train"
            ]
        self.context["training_dataset_size"] = self.training_data.shape[0]
        self.context["num_runs_in_experiment"] = mlflow.search_runs(
            experiment_ids=[self.experiment_id]
        ).shape[0]

    def collect_metadata(self):
        metadata_functions = [
            self.get_basic_context,
            self.get_feature_metadata,
            self.get_model_plots,
            self.section_registry.render_all,
        ]

        for func in metadata_functions:
            LOGGER.info(f"Updating context from {func.__name__}()")
            self.context.update(func())

    def get_basic_context(self) -> dict[str, str]:
        return {
            "logo": utils.download_static_asset(
                        description="Logo",
                        static_path=self.logo_path,
                        width=250,
                        local_folder=self.assets_folder
                    ),
            "institution_name": self.cfg.institution_name,
            "current_year": datetime.now().year,
        }

    def get_feature_metadata(self) -> dict[str, str]:
        feature_count = len(self.model.named_steps["column_selector"].get_params()["cols"])
        fs_cfg = self.cfg.modeling.feature_selection
        return {
            "number_of_features": feature_count,
            "collinearity_threshold": fs_cfg.collinear_threshold,
            "low_variance_threshold": fs_cfg.low_variance_threshold,
            "incomplete_threshold": fs_cfg.incomplete_threshold,
        }

    def get_model_plots(self) -> dict[str, str]:
        plots = {
            "model_comparison_plot": ("Model Comparison", "model_comparison.png", 450),
            "test_calibration_curve": ("Test Calibration Curve", "calibration/test_calibration.png", 475),
            "test_roc_curve": ("Test ROC Curve", "test_roc_curve_plot.png", 500),
            "test_confusion_matrix": ("Test Confusion Matrix", "test_confusion_matrix.png", 425),
            "test_histogram": ("Test Histogram", "preds/test_hist.png", 475),
            "feature_importances_by_shap_plot": ("Feature Importances", "shap_summary_labeled_dataset_100_ref_rows.png", 500),
        }
        return {
            key: utils.download_artifact(
                run_id=self.run_id, 
                description=description,
                artifact_path=path,
                width=width,
                local_folder=self.assets_folder
            )
            for key, (description, path, width) in plots.items()
        }

    def render(self):
        with open(self.template_path, "r") as file:
            template = file.read()
        filled = template.format(**self.context)
        with open(self.output_path, "w") as file:
            file.write(filled)
        LOGGER.info("✅ Model card generated!")
    
    def _extract_model_name(self, uc_model_name) -> str:
        return uc_model_name.split('.')[-1]

    def _build_output_path(self) -> str:
        filename = f"model-card-{self.model_name}.md"
        return os.path.join(os.getcwd(), filename)

    def _resolve_template(self, filename) -> importlib.abc.Traversable:
        return files("student_success_tool.reporting.template").joinpath(filename)

    def _resolve_asset(self, filename) -> importlib.abc.Traversable:
        return files("student_success_tool.reporting.template.assets").joinpath(filename)

    def _register_sections(self):
        register_sections(self, self.section_registry)


