import os
import logging
import typing as t

from mlflow.tracking import MlflowClient

# export .md to .pdf
import markdown
from weasyprint import HTML

# resolving files in templates module within package
from importlib.abc import Traversable
from importlib.resources import files

# internal SST modules
from ... import dataio, modeling

# relative imports in 'reporting' module
from ..sections import register_sections
from ..sections.registry import SectionRegistry
from ..utils import utils
from ..utils.formatting import Formatting
from ..utils.types import ModelCardConfig

LOGGER = logging.getLogger(__name__)
C = t.TypeVar("C", bound=ModelCardConfig)


class ModelCard(t.Generic[C]):
    DEFAULT_ASSETS_FOLDER = "card_assets"
    TEMPLATE_FILENAME = "model-card-TEMPLATE.md"
    LOGO_FILENAME = "logo.png"

    def __init__(
        self,
        config: C,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes the ModelCard object with the given config and the model name
        in unity catalog. If assets_path is not provided, the default assets folder is used.
        """
        self.cfg = config
        self.catalog = catalog
        self.model_name = model_name
        self.uc_model_name = f"{catalog}.{self.cfg.institution_id}_gold.{model_name}"
        LOGGER.info("Initializing ModelCard for model: %s", self.uc_model_name)

        self.client = mlflow_client or MlflowClient()
        self.section_registry = SectionRegistry()
        self.format = Formatting()
        self.context: dict[str, t.Any] = {}

        self.assets_folder = assets_path or self.DEFAULT_ASSETS_FOLDER
        self.output_path = self._build_output_path()
        self.template_path = self._resolve(
            "student_success_tool.reporting.template", self.TEMPLATE_FILENAME
        )
        self.logo_path = self._resolve(
            "student_success_tool.reporting.template.assets", self.LOGO_FILENAME
        )

    def build(self):
        """
        Builds the model card by performing the following steps:
        1. Loads the MLflow model.
        2. Finds the model version from the MLflow client based on the run ID.
        3. Extracts the training data from the MLflow run.
        4. Registers all sections in the section registry.
        5. Collects all metadata for the model card.
        6. Renders the model card using the template and context.
        """
        self.load_model()
        self.find_model_version()
        self.extract_training_data()
        self._register_sections()
        self.collect_metadata()
        self.render()

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

        self.model = dataio.models.load_mlflow_model(
            model_cfg.mlflow_model_uri,
            model_cfg.framework,
        )
        self.run_id = model_cfg.run_id
        self.experiment_id = model_cfg.experiment_id

    def find_model_version(self):
        """
        Retrieves the model version from the MLflow client based on the run ID.
        """
        try:
            versions = self.client.search_model_versions(f"name='{self.uc_model_name}'")
            for v in versions:
                if v.run_id == self.run_id:
                    self.context["version_number"] = v.version
                    LOGGER.info(f"Model Version = {self.context['version_number']}")
                    return
            LOGGER.warning(f"Unable to find model version for run id: {self.run_id}")
            self.context["version_number"] = None
        except Exception as e:
            LOGGER.error(
                f"Error retrieving model version for run id {self.run_id}: {e}"
            )
            self.context["version_number"] = None

    def extract_training_data(self):
        """
        Extracts the training data from the MLflow run utilizing SST internal subpackages (modeling).
        """
        self.modeling_data = modeling.evaluation.extract_training_data_from_model(
            self.experiment_id
        )
        self.training_data = self.modeling_data
        if self.cfg.split_col:
            self.training_data = self.modeling_data[
                self.modeling_data[self.cfg.split_col] == "train"
            ]
        self.context["training_dataset_size"] = self.training_data.shape[0]
        self.context["num_runs_in_experiment"] = utils.safe_count_runs(
            self.experiment_id
        )

    def collect_metadata(self):
        """
        Gathers all metadata for the model card. All of this data is dynamic and will
        depend on the institution and model. This calls functions that retrieves & downloads
        mlflow artifacts and also retrieves config information.
        """
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
        """
        Collects "basic" context which instantiates the DataKind logo, the
        institution name, and the current year.

        Returns:
            A dictionary with the keys as the variable names that will be called
            dynamically in template with values for each variable.
        """
        return {
            "logo": utils.download_static_asset(
                description="Logo",
                static_path=self.logo_path,
                local_folder=self.assets_folder,
            )
            or "",
            "institution_name": self.cfg.institution_name,
        }

    def get_feature_metadata(self) -> dict[str, str]:
        """
        Collects feature count from the MLflow run. Also, collects feature selection data
        from the config file.

        Returns:
            A dictionary with the keys as the variable names that will be called
            dynamically in template with values for each variable.
        """
        feature_count = len(
            self.model.named_steps["column_selector"].get_params()["cols"]
        )
        if not self.cfg.modeling or not self.cfg.modeling.feature_selection:
            raise ValueError(
                "Modeling configuration or feature selection config is missing."
            )

        fs_cfg = self.cfg.modeling.feature_selection

        return {
            "number_of_features": str(feature_count),
            "collinearity_threshold": str(fs_cfg.collinear_threshold),
            "low_variance_threshold": str(fs_cfg.low_variance_threshold),
            "incomplete_threshold": str(fs_cfg.incomplete_threshold),
        }

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
                "feature_importances_by_shap_plot.png",
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

    def render(self):
        """
        Renders the model card using the template and context data.
        """
        with open(self.template_path, "r") as file:
            template = file.read()
        filled = template.format(**self.context)
        with open(self.output_path, "w") as file:
            file.write(filled)
        LOGGER.info(f"✅ Model card generated at {self.output_path}")

    def reload_card(self):
        """
        Reloads Markdown model card post user editing after rendering.
        This offers flexibility in case user wants to utilize this class
        as a base and then makes edits in markdown before exporting as a PDF.
        """
        # Read the Markdown output
        with open(self.output_path, "r") as f:
            self.md_content = f.read()
        LOGGER.info("Reloaded model card content")

    def style_card(self):
        """
        Styles card using CSS.
        """
        # Convert Markdown to HTML
        html_content = markdown.markdown(
            self.md_content,
            extensions=["extra", "tables", "sane_lists", "toc", "smarty"],
        )

        # Load CSS from external file
        css_path = self._resolve(
            "student_success_tool.reporting.template.styles", "model_card.css"
        )
        with open(css_path, "r") as f:
            style = f"<style>\n{f.read()}\n</style>"

        # Prepend CSS to HTML
        self.html_content = style + html_content
        LOGGER.info("Applied CSS styling")

    def export_to_pdf(self):
        """
        Export CSS styled HTML to PDF utilizing weasyprint for conversion.
        Also logs the card, so it can be accessed as a PDF in the run artifacts.
        """
        # Styles card using model_card.css
        self.style_card()

        # Define PDF path
        base_path = os.path.dirname(self.output_path) or "."
        self.pdf_path = self.output_path.replace(".md", ".pdf")

        # Render PDF
        try:
            HTML(string=self.html_content, base_url=base_path).write_pdf(self.pdf_path)
            LOGGER.info(f"✅ PDF model card saved to {self.pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to create PDF: {e}")

        # Log card as an ML artifact
        utils.log_card(local_path=self.pdf_path, run_id=self.run_id)

    def _build_output_path(self) -> str:
        """
        Builds the output path for the model card.
        """
        filename = f"model-card-{self.model_name}.md"
        return os.path.join(os.getcwd(), filename)

    def _register_sections(self):
        """
        Registers all sections in the section registry.
        """
        register_sections(self, self.section_registry)

    def _resolve(self, package: str, filename: str) -> Traversable:
        """
        Resolves files using importlib. Importlib is necessary since
        the file exists within the SST package itself.
        """
        return files(package).joinpath(filename)
