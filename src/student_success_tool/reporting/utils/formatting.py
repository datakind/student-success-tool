class Formatting:
    def __init__(self, base_spaces: int = 2):
        """
        Initialize the formatter with a base indentation size.

        Args:
            base_spaces (int): The number of spaces for each indent level. Default is 2.
        """
        self.base_spaces = base_spaces

    def indent_level(self, depth: int) -> str:
        """
        Generate a string of spaces for indentation.
        """
        return " " * (self.base_spaces * depth)

    def header_level(self, depth: int) -> str:
        """
        Generate Markdown header prefix based on depth.
        """
        return "#" * depth

    def bold(self, text: str) -> str:
        """
        Apply Markdown bold formatting to a given text.
        """
        return f"**{text}**"

    def italic(self, text: str) -> str:
        """
        Apply Markdown italic formatting to a given text.
        """
        return f"_{text}_"
