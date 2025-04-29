class Formatting:
    def __init__(self, base_spaces: int = 2):
        self.base_spaces = base_spaces

    def indent_level(self, depth: int) -> str:
        return " " * (self.base_spaces * depth)
    
    def header_level(self, depth: int) -> str:
        return "#" * depth
    
    def bold(self, text: str) -> str:
        return f"**{text}**"
    
    def italic(self, text: str) -> str:
        return f"_{text}_"
    
