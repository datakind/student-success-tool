import json
import os
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from student_success_tool.ingestion_validation.validation import (
    normalize_col,
    load_json,
    merge_model_columns,
    build_schema,
    validate_dataset,
    HardValidationError,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

@ pytest.fixture
def base_schema_file(tmp_path):
    """
    Create a minimal base_schema.json with one model ('student')
    including both required and optional columns.
    """
    schema = {
        "base": {
            "data_models": {
                "student": {
                    "columns": {
                        "student_id": {
                            "dtype": "string",
                            "coerce": True,
                            "nullable": False,
                            "required": True,
                            "aliases": ["guid", "study_id"],
                            "checks": [{"type": "str_length", "args": [], "kwargs": {"min_value": 3}}]
                        },
                        "age": {
                            "dtype": "string",
                            "coerce": True,
                            "nullable": False,
                            "required": False,
                            "aliases": [],
                            "checks": [{"type": "str_length", "args": [], "kwargs": {"min_value": 2}}]
                        },
                        "disability_status": {
                            "dtype": "string",
                            "coerce": True,
                            "nullable": True,
                            "required": False,
                            "aliases": [],
                            "checks": []
                        }
                    }
                }
            }
        }
    }
    p = tmp_path / "base_schema.json"
    p.write_text(json.dumps(schema))
    return str(p)

@ pytest.fixture
def ext_schema_file(tmp_path):
    """
    Create an extension schema adding one optional column under 'student'.
    """
    ext = {
        "institutions": {
            "institution": {
                "data_models": {
                    "student": {
                        "columns": {
                            "enrollment_type": {
                                "dtype": "string",
                                "coerce": True,
                                "nullable": True,
                                "required": False,
                                "aliases": ["enroll_type"],
                                "checks": []
                            }
                        }
                    }
                }
            }
        }
    }
    p = tmp_path / "ext_schema.json"
    p.write_text(json.dumps(ext))
    return str(p)

# ─── Tests for utility functions ─────────────────────────────────────────────

def test_normalize_col():
    assert normalize_col("  Foo-Bar Baz ") == "foo_bar_baz"
    assert normalize_col("MIXED case") == "mixed_case"


def test_load_json_success(tmp_path):
    data = {"a": 1}
    p = tmp_path / "file.json"
    p.write_text(json.dumps(data))
    assert load_json(str(p)) == data


def test_load_json_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_json(str(tmp_path / "missing.json"))

# malformed JSON also raises FileNotFoundError

def test_load_json_malformed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(FileNotFoundError):
        load_json(str(bad))

# merge_model_columns

def test_merge_model_columns_base_only(base_schema_file):
    base = json.load(open(base_schema_file))
    merged = merge_model_columns(base, extension_schema=None, institution="institution", model="student")
    assert set(merged) == {"student_id", "age", "disability_status"}


def test_merge_model_columns_with_extension(base_schema_file, ext_schema_file):
    base = json.load(open(base_schema_file))
    ext = json.load(open(ext_schema_file))
    merged = merge_model_columns(base, ext, "institution", "student")
    assert set(merged) == {"student_id", "age", "disability_status", "enrollment_type"}


def test_merge_model_columns_missing_model(base_schema_file):
    base = json.load(open(base_schema_file))
    with pytest.raises(KeyError):
        merge_model_columns(base, None, "institution", "unknown_model")

# build_schema

def test_build_schema_and_validate(base_schema_file):
    df = pd.DataFrame({
        "student_id": ["ABC123", "DEF456"],
        "age": ["21+", "<21"],
        "disability_status": ["Y", None],
    })
    base = json.load(open(base_schema_file))
    specs = base["base"]["data_models"]["student"]["columns"]
    schema = build_schema(specs)
    validated = schema.validate(df)
    pd.testing.assert_frame_equal(validated, df, check_dtype=False)

# ─── Tests for validate_dataset ─────────────────────────────────────────────

def test_validate_dataset_extra_columns(tmp_path, base_schema_file):
    df = pd.DataFrame({
        "student_id": ["ABC123"],
        "age": ["21+"],
        "disability_status": ["Y"],
        "unexpected": [1],
    })
    with pytest.raises(HardValidationError) as exc:
        validate_dataset(
            df,
            models="student",
            institution_id="pdp",
            base_schema_path=base_schema_file,
            extension_schema_path=None,
        )
    err = exc.value
    assert err.extra_columns == ["unexpected"]
    assert err.missing_required == []


def test_validate_dataset_missing_required(tmp_path, base_schema_file):
    df = pd.DataFrame({
        "age": ["21+"],
        "disability_status": ["Y"],
    })
    with pytest.raises(HardValidationError) as exc:
        validate_dataset(
            df,
            models="student",
            institution_id="institution",
            base_schema_path=base_schema_file,
            extension_schema_path=None,
        )
    err = exc.value
    assert err.missing_required == ["student_id"]


def test_validate_dataset_schema_errors(tmp_path, base_schema_file):
    # str_length min 3, provide too short
    df = pd.DataFrame({
        "student_id": ["AB"],
        "age": ["21+"],
        "disability_status": ["Y"],
    })
    with pytest.raises(HardValidationError) as exc:
        validate_dataset(
            df,
            models="student",
            institution_id="institution",
            base_schema_path=base_schema_file,
            extension_schema_path=None,
        )
    err = exc.value
    assert err.schema_errors is not None
    assert isinstance(err.schema_errors, list)


def test_validate_dataset_soft_pass(tmp_path, base_schema_file, ext_schema_file):
    # omit optional 'age' and ext optional 'enrollment_type'
    df = pd.DataFrame({"student_id": ["ABC123"], "disability_status": ["N"]})
    result = validate_dataset(
        df,
        models="student",
        institution_id="institution",
        base_schema_path=base_schema_file,
        extension_schema_path=ext_schema_file,
    )
    assert result["validation_status"] == "passed_with_soft_errors"
    assert set(result["missing_optional"]) == {"age", "enrollment_type"}


def test_validate_dataset_csv_input(tmp_path, base_schema_file):
    df = pd.DataFrame({"student_id": ["ABC123"], "age": ["21+"], "disability_status": ["N"]})
    csv = tmp_path / "in.csv"
    df.to_csv(csv, index=False)
    result = validate_dataset(
        str(csv),
        models="student",
        institution_id="institution",
        base_schema_path=base_schema_file,
    )
    assert result["validation_status"] in ("passed", "passed_with_soft_errors")
