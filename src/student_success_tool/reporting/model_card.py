import os
import shutil
import mlflow
import logging
from mlflow.tracking import MlflowClient
from datetime import datetime
from importlib.resources import files

# internal SST modules
from .. import dataio, modeling

from .sections import registry, register_sections
from .sections.registry import SectionRegistry

LOGGER = logging.getLogger(__name__)

class ModelCard:
    def __init__(self, config, uc_model_name):
        self.cfg = config
        self.uc_model_name = uc_model_name
        self.model_name = self.uc_model_name.split('.')[-1]
        self.client = MlflowClient()
        self.context = {}
        self.output_path = os.path.join(os.getcwd(), f"model-card-{self.model_name}.md")
        self.template_path = self._resolve_template("model-card-TEMPLATE.md")
        self.logo_path = self._resolve_asset("logo.png")
        self.section_registry = SectionRegistry()
        self._register_sections()

    def build(self):
        self.load_model()
        self.find_model_version()
        self.extract_training_data()
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

    def get_basic_context(self):
        return {
            "logo": self.download_static_asset("Logo", self.logo_path, width=250),
            "institution_name": self.cfg.institution_name,
            "current_year": datetime.now().year,
        }

    def get_feature_metadata(self):
        feature_count = len(self.model.named_steps["column_selector"].get_params()["cols"])
        fs_cfg = self.cfg.modeling.feature_selection
        return {
            "number_of_features": feature_count,
            "collinearity_threshold": fs_cfg.collinear_threshold,
            "low_variance_threshold": fs_cfg.low_variance_threshold,
            "incomplete_threshold": fs_cfg.incomplete_threshold,
        }

    def get_model_plots(self):
        plots = {
            "model_comparison_plot": ("Model Comparison", "model_comparison.png", 400),
            "test_calibration_curve": ("Test Calibration Curve", "calibration/test_calibration.png", 400),
            "test_roc_curve": ("Test ROC Curve", "test_roc_curve_plot.png", 400),
            "test_confusion_matrix": ("Test Confusion Matrix", "test_confusion_matrix.png", 400),
            "test_histogram": ("Test Histogram", "preds/test_hist.png", 400),
            "feature_importances_by_shap_plot": ("Feature Importances", "shap_summary_labeled_dataset_100_ref_rows.png", 400),
        }
        return {
            key: self.download_artifact(description, path, width)
            for key, (description, path, width) in plots.items()
        }

    def download_artifact(self, description, artifact_path, width, local_folder="artifacts"):
        os.makedirs(local_folder, exist_ok=True)
        local_path = mlflow.artifacts.download_artifacts(
            run_id=self.run_id,
            artifact_path=artifact_path,
            dst_path=local_folder,
        )
        return self.embed_image(description, local_path, width)

    def download_static_asset(self, description, static_path, width, local_folder="artifacts"):
        artifacts_dir = os.path.join(self.output_dir, local_folder)
        os.makedirs(artifacts_dir, exist_ok=True)

        dst_path = os.path.join(artifacts_dir, static_path.name)
        shutil.copy(static_path, dst_path)

        return self.embed_image(description, dst_path, width)

    def embed_image(self, description, local_path, width):
        return f'<img src="{os.path.relpath(local_path, start=os.getcwd())}" alt="{description}" width="{width}">'
    
    def render(self):
        with open(self.template_path, "r") as file:
            template = file.read()
        filled = template.format(**self.context)
        with open(self.output_path, "w") as file:
            file.write(filled)
        LOGGER.info("✅ Model card generated!")
    
    def _resolve_template(self, filename):
        return files("student_success_tool.reporting.template").joinpath(filename)

    def _resolve_asset(self, filename):
        return files("student_success_tool.reporting.template.assets").joinpath(filename)

    def _register_sections(self):
        register_sections(self, self.section_registry)

    @property
    def output_dir(self):
        return os.path.dirname(os.path.abspath(self.output_path))

