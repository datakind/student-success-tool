class SectionRegistry:
    def __init__(self):
        self._sections = []

    def register(self, key):
        def decorator(fn):
            self._sections.append((key, fn))
            return fn
        return decorator

    def render_all(self):
        return {key: fn() for key, fn in self._sections}