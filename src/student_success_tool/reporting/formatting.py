class Indentation:
    def __init__(self, base_spaces: int = 2):
        self.base_spaces = base_spaces

    def level(self, depth: int) -> str:
        return " " * (self.base_spaces * depth)