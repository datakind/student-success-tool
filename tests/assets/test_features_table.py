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
def test_feature_matches_exactly_one_regex_key(feature_name, feature_table_data):
    """Ensure each valid feature name matches exactly one regex key from the TOML."""

    def is_likely_regex(key: str) -> bool:
        # Matches if the key contains metacharacters indicating it's a regex
        return key.startswith("^") or bool(re.search(r"[\(\[\.\*\+\?\\]", key))

    # Only consider keys with escape sequences or regex metacharacters
    regex_keys = [key for key in feature_table_data.keys() if is_likely_regex(key)]

    # Compile the regex patterns
    compiled_patterns = [re.compile(pat) for pat in regex_keys]

    matches = [pat.pattern for pat in compiled_patterns if pat.fullmatch(feature_name)]

    assert len(matches) == 1, (
        f"Feature '{feature_name}' matched {len(matches)} regex patterns: {matches}. "
        "Expected to match exactly one."
    )
