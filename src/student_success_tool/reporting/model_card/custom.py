import typing as t
from mlflow.tracking import MlflowClient

from ...configs.custom import CustomProjectConfig
from .base import ModelCard
from ..sections.custom import register_sections as register_custom_sections


class CustomModelCard(ModelCard[CustomProjectConfig]):
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
