import pytest
import re
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


# Add regex test cases to this as needed! This helps us make sure
# that are regex patterns work *as intended* in our features table
VALID_FEATURE_NAMES = [
    "took_course_id_eng_101",
    "took_course_id_eng_101_cummax",
    "took_course_id_eng_101_cummax_in_12_creds",
    "took_course_subject_area_51",
    "took_course_subject_area_51_cummax",
    "took_course_subject_area_51_cummax_in_12_creds",
    "num_courses_course_subject_area_51",
    "num_courses_course_subject_area_51_cumfrac",
    "frac_courses_course_subject_area_51",
    "num_courses_course_id_eng_101",
    "num_courses_course_id_eng_101_cumfrac",
    "frac_courses_course_id_eng_101",
    "took_course_id_english_composition_and_writing_098_cummax"
]


@pytest.mark.parametrize("feature_name", VALID_FEATURE_NAMES)
def test_feature_matches_some_regex_key(feature_name, feature_table_data):
    """Check if each valid feature name matches at least one of the TOML regex keys."""

    def is_likely_regex(key: str) -> bool:
        # Matches if the key contains metacharacters indicating it's a regex
        return bool(re.search(r"[\(\[\.\*\+\?\\]", key))

    # Only consider keys with \d in them — implying dynamic regex patterns
    regex_keys = [key for key in feature_table_data.keys() if is_likely_regex(key)]

    # Compile the regex patterns
    compiled_patterns = [re.compile(pattern) for pattern in regex_keys]

    # Check if the feature_name matches ANY of them
    matched = any(pat.fullmatch(feature_name) for pat in compiled_patterns)

    assert matched, (
        f"Feature '{feature_name}' did not match any regex pattern with \\d in the TOML"
    )
