import os
import pathlib
from contextlib import nullcontext as does_not_raise

import pydantic as pyd
import pytest

from student_success_tool import dataio

try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa


FIXTURES_PATH = "tests/fixtures"


class BadProjectConfig(pyd.BaseModel):
    is_bad: bool = pyd.Field(...)


@pytest.mark.parametrize(
    ["file_name", "schema", "context"],
    [
        # TODO
        # ("project_config.toml", PDPProjectConfig, does_not_raise()),
        ("project_config.toml", BadProjectConfig, pytest.raises(pyd.ValidationError)),
    ],
)
def test_load_config(file_name, schema, context):
    file_path = os.path.join(FIXTURES_PATH, file_name)
    with context:
        result = dataio.read_config(file_path, schema=schema)
        assert isinstance(result, pyd.BaseModel)


@pytest.mark.parametrize(
    "toml_content, expected_output, expect_exception",
    [
        (
            """
            academic_term = { name = "academic term" }
            term_in_peak_covid = { name = "term occurred in 'peak' COVID" }
            num_courses = { name = "number of courses taken this term" }
            """,
            {
                "academic_term": {"name": "academic term"},
                "term_in_peak_covid": {"name": "term occurred in 'peak' COVID"},
                "num_courses": {"name": "number of courses taken this term"},
            },
            does_not_raise(),
        ),
        (
            """
            academic_term = { name = "academic term" }
            term_in_peak_covid = { name = "term occurred in 'peak' COVID" }
            num_courses = { name = "number of courses taken this term"
            """,
            None,
            pytest.raises(tomllib.TOMLDecodeError),
        ),
        (
            "",
            None,
            pytest.raises(FileNotFoundError),
        ),
    ],
)
def test_read_features_table(tmpdir, toml_content, expected_output, expect_exception):
    if toml_content:
        toml_file = tmpdir.join("features_table.toml")
        toml_file.write(toml_content)
        file_path = str(toml_file)
    else:
        file_path = "non_existent_path/features_table.toml"
    with expect_exception:
        features_table = dataio.read_features_table(file_path)
        if expect_exception is does_not_raise():
            assert isinstance(features_table, dict)
            assert features_table == expected_output


class MinimalProjectConfig(pyd.BaseModel):
    model: dict


def test_write_config_success(tmp_path):
    # Given
    config = MinimalProjectConfig(model={"run_id": "abc123", "experiment_id": "xyz789"})
    file_path = tmp_path / "test_config.toml"

    # When
    dataio.write_config(config, str(file_path))

    # Then
    assert file_path.exists()
    contents = file_path.read_text()
    assert 'run_id = "abc123"' in contents
    assert 'experiment_id = "xyz789"' in contents


def test_write_config_failure(monkeypatch):
    # Given
    config = MinimalProjectConfig(model={"run_id": "abc123"})
    fake_path = "/invalid/path/config.toml"

    def raise_os_error(*args, **kwargs):
        raise OSError("Mocked write failure")

    monkeypatch.setattr(pathlib.Path, "write_text", raise_os_error)

    # Then
    with pytest.raises(OSError, match="Failed to write config"):
        dataio.write_config(config, fake_path)
