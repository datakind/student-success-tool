import pytest
import os
from student_success_tool.dataio.read import from_toml_file


@pytest.fixture
def feature_table_data():
    project_root = os.getcwd()
    toml_path = os.path.join(
        project_root,
        "src",
        "student_success_tool",
        "assets",
        "pdp",
        "features_table.toml",
    )
    return from_toml_file(toml_path)


def test_toml_file_loads(feature_table_data):
    assert isinstance(feature_table_data, dict)
    assert len(feature_table_data) > 0, "The TOML file appears to be empty."


def test_all_features_have_name_and_desc(feature_table_data):
    for feature_id, entry in feature_table_data.items():
        assert isinstance(entry, dict), f"Entry for {feature_id} should be a dict."
        assert "name" in entry and entry["name"].strip(), (
            f"'name' missing or empty in feature: {feature_id}"
        )
        assert "desc" in entry and entry["desc"].strip(), (
            f"'desc' missing or empty in feature: {feature_id}"
        )
