import typing as t
from mlflow.tracking import MlflowClient

from ...configs.pdp import PDPProjectConfig
from .base import ModelCard
from ..sections.pdp import register_sections as register_pdp_sections


class PDPModelCard(ModelCard[PDPProjectConfig]):
    def __init__(
        self,
        config: PDPProjectConfig,
        catalog: str,
        model_name: str,
        assets_path: t.Optional[str] = None,
        mlflow_client: t.Optional[MlflowClient] = None,
    ):
        """
        Initializes PDP model card by enforcing a PDP project config.
        Otherwise, this class inherits and is functionally the same as the
        base ModelCard class.
        """
        if not isinstance(config, PDPProjectConfig):  # type guard
            raise TypeError("Expected config to be of type PDPProjectConfig")

        super().__init__(config, catalog, model_name, assets_path, mlflow_client)

    def _register_sections(self):
        """
        Register PDP-specific sections.
        """
        # Clearing registry for overrides
        self.section_registry.clear()

        # Register PDP-specific sections
        register_pdp_sections(self, self.section_registry)
