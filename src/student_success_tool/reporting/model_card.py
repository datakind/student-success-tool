import os
from datetime import datetime
from mlflow.tracking import MlflowClient
import mlflow

from .sections import registry, register_sections
from .sections.registry import SectionRegistry

package_dir = os.path.dirname(os.path.abspath(__file__))

class ModelCard:
    def __init__(self, config, uc_model_name):
        self.cfg = config
        self.uc_model_name = uc_model_name
        self.model_name = self.uc_model_name.split('.')[-1]
        self.template_path = os.path.join(package_dir, "..", "templates", "model-card-TEMPLATE.md")
        self.output_path = os.path.join(os.getcwd(), f"model-card-{self.model_name}.md")        self.client = MlflowClient()
        self.context = {}
        self.section_registry = SectionRegistry()
        self.register_sections()

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
        self.context.update({
            "institution_name": self.cfg.institution_name,
            "current_year": datetime.now().year,
            **self.get_feature_selection_metadata(),
            "number_of_features": self.get_feature_count(),
            "model_comparison_plot": self.download_artifact("Model Comparison", "model_comparison.png"),
            **self.section_registry.render_all()
        })

    def get_feature_selection_metadata(self):
        fs_cfg = self.cfg.modeling.feature_selection
        return {
            "collinearity_threshold": fs_cfg.collinear_threshold,
            "low_variance_threshold": fs_cfg.low_variance_threshold,
            "incomplete_threshold": fs_cfg.incomplete_threshold
        }

    def get_feature_count(self):
        return len(self.model.named_steps["column_selector"].get_params()["cols"])

    def download_artifact(self, description, artifact_path, local_folder="artifacts"):
        os.makedirs(local_folder, exist_ok=True)
        local_path = mlflow.artifacts.download_artifacts(
            run_id=self.run_id,
            artifact_path=artifact_path,
            dst_path=local_folder,
        )
        return f"![{description}]({os.path.relpath(local_path, start=os.getcwd())})"

    def render(self):
        with open(self.template_path, "r") as file:
            template = file.read()
        filled = template.format(**self.context)
        with open(self.output_path, "w") as file:
            file.write(filled)
        print("✅ Model card generated!")
