import logging
import typing as t

LOGGER = logging.getLogger(__name__)


class SectionRegistry:
    """
    Collects and manages all registered model card sections.

    A SectionRegistry object allows section-generating functions to be registered under specific
    keys (e.g., "outcome_section") and later rendered into a markdown-friendly format. This
    design supports modularity, scale, and unit testing of individual sections of the model card.
    """

    def __init__(self):
        self._sections = []

    def register(
        self,
        key: str,
    ) -> t.Callable[[t.Callable[[], str]], t.Callable[[], str]]:
        """
        Registers a section-rendering function under a specific key.

        This is used as a decorator to attach a section generator (e.g. outcome section,
        bias table, etc.) to the registry. Each registered function is stored along with its
        key and later rendered via `render_all()`.

        Args:
            key: The identifier used to reference the section in the model card template.

        Returns:
            A decorator that registers the section-generating function.
        """

        def decorator(fn: t.Callable[[], str]) -> t.Callable[[], str]:
            self._sections.append((key, fn))
            return fn

        return decorator

    def render_all(self) -> dict[str, str]:
        """
        Renders all registered sections into a dictionary for markdown formatting.

        Returns:
            A mapping of section keys to their rendered markdown strings.
        """
        return {key: fn() for key, fn in self._sections}

    def clear(self) -> None:
        """
        Clears all registered sections for platform-specific processing & overrides.
        """
        self._sections.clear()
