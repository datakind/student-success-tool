import re

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
    
    def friendly_case(text: str, capitalize: bool = True) -> str:
        """
        Converts strings like 'bachelor's degree' or 'full-time' into human-friendly forms,
        preserving hyphens and apostrophes, with optional capitalization.

        Args:
            text (str): Raw string (e.g., from config)
            capitalize (bool): Whether to title-case each word. If False, keeps original casing.

        Returns:
            str: Human-friendly string.
        """
        text = text.replace("_", " ")

        def smart_cap(word: str) -> str:
            # Handles hyphenated subwords like "full-time"
            return "-".join(
                part[0].upper() + part[1:].lower() if part else ""
                for part in word.split("-")
            )

        if not capitalize:
            return text

        # Regex preserves apostrophes and hyphens
        tokens = re.findall(r"[\\w'-]+", text)
        return " ".join(smart_cap(tok) for tok in tokens)


