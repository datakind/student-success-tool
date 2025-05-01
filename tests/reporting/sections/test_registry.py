from student_success_tool.reporting.sections.registry import SectionRegistry


def test_register_and_render_single_section():
    registry = SectionRegistry()

    @registry.register("test_section")
    def test_fn():
        return "This is a test section."

    rendered = registry.render_all()
    assert rendered == {"test_section": "This is a test section."}


def test_register_and_render_multiple_sections():
    registry = SectionRegistry()

    @registry.register("section_one")
    def one():
        return "Section One"

    @registry.register("section_two")
    def two():
        return "Section Two"

    result = registry.render_all()
    assert result["section_one"] == "Section One"
    assert result["section_two"] == "Section Two"
    assert len(result) == 2
