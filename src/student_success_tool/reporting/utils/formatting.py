import re
import typing as t


class Formatting:
    def __init__(self, base_spaces: int = 4):
        """
        Initialize the formatter with a base indentation size.

        Args:
            base_spaces: The number of spaces for each indent level. The default
            needs to be 4, since for markdown parsers and PDF export, this would
            create a reliable interpretation of nested lists.
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

    def ordinal(self, n: int) -> str:
        """
        Converts an integer to its ordinal form (e.g. 1 -> 1st).
        """
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def friendly_case(self, text: str, capitalize: bool = True) -> str:
        """
        Converts strings like 'bachelor's degree' or 'full-time' into human-friendly forms,
        preserving hyphens and apostrophes, with optional capitalization.

        Args:
            text: Text to be converted
            capitalize: Whether to title-case each word. If False, keeps original casing.

        Returns:
            Human-friendly string.
        """
        if isinstance(text, (int, float)):
            return str(text)

        # If the string is numeric-like (int or float), return as-is
        try:
            float_val = float(text)
            if text.strip().replace(".", "", 1).isdigit() or text.strip().isdigit():
                return text
        except ValueError:
            pass  # Not a float-like string; continue formatting

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
        tokens = re.findall(r"[\w'-]+", text)
        return " ".join(smart_cap(tok) for tok in tokens)

    def format_intensity_time_limit(self, duration: t.Tuple[str, str]) -> str:
        """
        We want to format a intensity_limit within config.toml by unpacking
        the value (3.0) and unit ("year"), for example.

        Args:
            duration: intensity limit in config (3.0, "year"), for example.
        """
        num, unit = duration

        # Format number cleanly
        if isinstance(num, float):
            if num.is_integer():
                num = int(num)
            else:
                num = round(num, 2)

        unit = unit if num == 1 else unit + "s"
        return f"{num} {unit}"
