import pytest
import os
from student_success_tool.dataio.read import from_toml_file
import re 
from student_success_tool.modeling.inference import _get_mapped_feature_name


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
        assert "short_desc" in entry and entry["short_desc"].strip(), (
            f"'short_desc' missing or empty in feature: {feature_id}"
        )
        assert "long_desc" in entry and entry["long_desc"].strip(), (
            f"'long_desc' missing or empty in feature: {feature_id}"
        )


# Add regex test cases to this as needed! This helps us make sure
# that are regex patterns work *as intended* in our features table
VALID_FEATURE_NAMES = [
    "took_course_id_eng_101",
    "took_course_id_eng_101_cummax",
    "took_course_id_eng_101_cummax_in_12_creds",
    "took_course_subject_area_51",
    "took_course_subject_area_51_cummax",
    "took_course_subject_area_51_cummax_in_12_creds",
    "took_course_id_english_composition_and_writing_098",
    "took_course_id_english_composition_and_writing_098_cummax",
    "took_course_id_english_composition_and_writing_098_cummax_in_12_creds",
    "took_course_id_english_composition_and_writing_101_1",
    "took_course_id_english_composition_and_writing_101_1_cummax",
    "took_course_id_english_composition_and_writing_101_1_cummax_in_12_creds",
    "num_courses_course_subject_area_51",
    "num_courses_course_subject_area_51_cumfrac",
    "frac_courses_course_subject_area_51",
    "num_courses_course_id_eng_101",
    "num_courses_course_id_eng_101_cumfrac",
    "frac_courses_course_id_eng_101",
    "took_course_id_english1010",
    "took_course_id_english1010_cummax",
    "took_course_id_english1010_cummax_in_12_creds",
    "took_course_id_mathematics_science1210",
    "took_course_id_mathematics_science1210_cummax",
    "took_course_id_mathematics_science1210_cummax_in_12_creds",
    "took_course_id_computer_science_1050_cummax",
    "took_course_id_communication_studies_1010_cummax_in_12_creds",
    "took_course_id_computer_science_1050_cummax_in_12_creds",
    "num_courses_course_id_mathematics_science1210",
    "num_courses_course_id_english1010",
    "num_courses_course_id_english_composition_and_writing_098",
    "num_courses_course_id_english_composition_and_writing_101_1",
    "num_courses_course_id_mathematics_science1210_cumfrac",
    "num_courses_course_id_english1010_cumfrac",
    "num_courses_course_id_english_composition_and_writing_098_cumfrac",
    "num_courses_course_id_english_composition_and_writing_101_1_cumfrac",
    "frac_courses_course_id_mathematics_science1210",
    "frac_courses_course_id_english1010",
    "frac_courses_course_id_english_composition_and_writing_098",
    "frac_courses_course_id_english_composition_and_writing_101_1",
]


@pytest.mark.parametrize("feature_name", VALID_FEATURE_NAMES)
def test_feature_maps_to_named_entry(feature_name, feature_table_data):
    """Ensure each feature maps to a named entry in the feature table using the production mapping logic."""
    mapped = _get_mapped_feature_name(feature_name, feature_table_data)

    # If it mapped to something, it must not be the identity (fallback case)
    assert mapped != feature_name, (
        f"Feature '{feature_name}' was not mapped correctly using _get_mapped_feature_name."
    )

    # Ensure that the result is a formatted name string
    assert isinstance(mapped, str) and mapped.strip(), (
        f"Mapped name for '{feature_name}' is empty or invalid: {mapped}"
    )

    # Ensure only one regex pattern matches this feature
    matching_patterns = [
        pattern for pattern in feature_table_data
        if re.fullmatch(pattern, feature_name)
    ]
    assert len(matching_patterns) == 1, (
        f"Feature '{feature_name}' matches {len(matching_patterns)} patterns: {matching_patterns}. "
        "Feature should match exactly one pattern."
    )