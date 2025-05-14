import json
import pandas as pd
import pytest

from validate_dataset import (
    normalize_col,
    merge_model_columns,
    build_schema,
    validate_dataset,
    HardValidationError,
)
from generate_extensions import (
    load_json as load_ext_json,
    infer_column_schema,
    generate_extension_schema,
)


@pytest.fixture
def base_schema_file(tmp_path):
    """
    minimal base_schema.json with one model (student)
    with required and optional columns.
    """
    schema = {
        "version": "1.0.0",
        "base": {
            "data_models": {
                "student": {
                    "columns": {
                        "student_id": {
                            "dtype": "string",
                            "coerce": True,
                            "nullable": False,
                            "required": True,
                            "aliases": ["guid", "study_id", "student_guid"],
                            "checks": [
                                {
                                    "type": "str_length",
                                    "args": [],
                                    "kwargs": {"min_value": 3},
                                }
                            ],
                        },
                        "age": {
                            "dtype": "string",
                            "coerce": True,
                            "nullable": False,
                            "required": False,
                            "aliases": [],
                            "checks": [
                                {
                                    "type": "str_length",
                                    "args": [],
                                    "kwargs": {"min_value": 2},
                                }
                            ],
                        },
                        "disability_status": {
                            "dtype": "string",
                            "coerce": True,
                            "nullable": True,
                            "required": False,
                            "aliases": [],
                            "checks": [],
                        },
                    }
                }
            }
        },
    }
    p = tmp_path / "base_schema.json"
    p.write_text(json.dumps(schema))
    return str(p)


@pytest.fixture
def ext_schema_file(tmp_path):
    """
    Creates an extension schema adding one extra optional column.
    """
    ext = {
        "institutions": {
            "pdp": {
                "data_models": {
                    "student": {
                        "columns": {
                            "enrollment_type": {
                                "dtype": "string",
                                "coerce": True,
                                "nullable": True,
                                "required": False,
                                "aliases": ["enroll_type"],
                                "checks": [],
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


def test_normalize_col():
    assert normalize_col("  student_id ") == "student_id"
    assert normalize_col("MIXED case") == "mixed_case"


def test_merge_model_columns_base_only(base_schema_file):
    base = json.load(open(base_schema_file))
    merged = merge_model_columns(
        base, extension_schema=None, institution="pdp", model="student"
    )
    assert set(merged) == {"student_id", "age", "disability_status"}


def test_merge_model_columns_with_extension(base_schema_file, ext_schema_file):
    base = json.load(open(base_schema_file))
    ext = json.load(open(ext_schema_file))
    merged = merge_model_columns(base, ext, "pdp", "student")
    assert set(merged) == {"student_id", "age", "disability_status", "enrollment_type"}


def test_build_schema_and_simple_validation(tmp_path, base_schema_file):
    df = pd.DataFrame(
        {
            "student_id": ["AB459F7SD", "EF455F4SD", "GH428F4SD"],
            "age": ["U21", "21-35", "O35"],
            "disability_status": ["Y", "N", None],
        }
    )
    base = json.load(open(base_schema_file))
    specs = base["base"]["data_models"]["student"]["columns"]
    schema = build_schema(specs)
    validated = schema.validate(df)
    pd.testing.assert_frame_equal(validated, df)


def test_extra_columns_hard_error(tmp_path, base_schema_file):
    df = pd.DataFrame(
        {
            "student_id": ["AB459F7SD"],
            "age": ["U21"],
            "disability_status": ["Y"],
            "unexpected": ["x"],
        }
    )
    with pytest.raises(HardValidationError) as exc:
        validate_dataset(
            df, base_schema_file, "student", "pdp", extension_schema_path=None
        )
    err = exc.value
    assert "unexpected columns" in str(err).lower()
    assert err.extra_columns == ["unexpected"]
    assert err.missing_required == []


def test_missing_required_hard_error(tmp_path, base_schema_file):
    df = pd.DataFrame({"age": ["U21"], "disability_status": ["Y"]})
    with pytest.raises(HardValidationError) as exc:
        validate_dataset(
            df, base_schema_file, "student", "pdp", extension_schema_path=None
        )
    err = exc.value
    assert "missing required columns" in str(err).lower()
    assert err.missing_required == ["student_id"]


def test_schema_type_errors(tmp_path, base_schema_file):
    df = pd.DataFrame(
        {"student_id": ["AB"], "age": ["u21"], "disability_status": ["Y"]}
    )
    with pytest.raises(HardValidationError) as exc:
        validate_dataset(
            df, base_schema_file, "student", "pdp", extension_schema_path=None
        )
    err = exc.value
    assert err.schema_errors is not None
    assert isinstance(err.schema_errors, list)


def test_soft_pass(tmp_path, base_schema_file, ext_schema_file):
    df = pd.DataFrame({"student_id": ["ABC"], "disability_status": ["Y"]})
    result = validate_dataset(
        df,
        base_schema_file,
        "student",
        "pdp",
        extension_schema_path=ext_schema_file,
    )
    assert result["validation_status"] == "passed_with_soft_errors"
    assert set(result["missing_optional"]) == {"age", "enrollment_type"}


def test_csv_input(tmp_path, base_schema_file):
    df = pd.DataFrame(
        {"student_id": ["ABC"], "age": ["u21"], "disability_status": ["Y"]}
    )
    csv = tmp_path / "in.csv"
    df.to_csv(csv, index=False)
    result = validate_dataset(str(csv), base_schema_file, "student", "pdp")
    assert result["validation_status"] in ("passed", "passed_with_soft_errors")


# ─── Now tests for generate_extensions.py ─────────────────────────────────────


def test_ext_load_json_missing(tmp_path):
    # should return {} on missing or invalid JSON
    assert load_ext_json(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("not-json")
    assert load_ext_json(str(bad)) == {}


def test_infer_column_schema_category_and_numeric():
    import numpy as np

    s_cat = pd.Series(["A", "B", "A", None])
    spec_cat = infer_column_schema(s_cat, cate_threshold=5)
    assert spec_cat["dtype"] == "category"
    assert "A" in spec_cat["categories"]
    assert spec_cat["nullable"] is True

    s_num = pd.Series([0, 5, np.nan, 10])
    spec_num = infer_column_schema(s_num)
    assert spec_num["dtype"] == "float64"
    # should include a non-negative check
    assert any(chk["type"] in ("ge", "between") for chk in spec_num["checks"])


def test_generate_extension_no_extras(tmp_path, base_schema_file, capsys):
    df = pd.DataFrame({"student_id": ["ABC"]})
    out = generate_extension_schema(
        df,
        base_schema_file,
        "pdp",
        "student",
        extension_schema_path=None,
        output_dir=str(tmp_path),
    )
    captured = capsys.readouterr()
    assert out == {}
    assert "No extra columns to extend" in captured.out
    assert not (tmp_path / "pdp_extension_schema.json").exists()


def test_generate_extension_with_extras(tmp_path, base_schema_file):
    df = pd.DataFrame({"student_id": ["ABC"], "foo": [1.2], "bar": ["X"]})
    ext = generate_extension_schema(
        df,
        base_schema_file,
        "pdp",
        ["student"],
        extension_schema_path=None,
        output_dir=str(tmp_path),
    )
    # top‐level version should be carried over
    assert ext["version"] == "1.0.0"
    cols = ext["institutions"]["pdp"]["data_models"]["student"]["columns"]
    assert set(cols) == {"foo", "bar"}
    # file was written
    assert (tmp_path / "pdp_extension_schema.json").exists()
    # re‐loading it matches
    loaded = json.load(open(tmp_path / "pdp_extension_schema.json"))
    assert loaded == ext


def test_generate_extension_updates_existing(tmp_path, base_schema_file):
    # seed an existing extension file in output_dir
    existing = {
        "version": "9.9.9",
        "institutions": {
            "pdp": {
                "data_models": {
                    "student": {
                        "columns": {
                            "orig": {
                                "dtype": "string",
                                "coerce": True,
                                "nullable": False,
                                "required": False,
                                "aliases": [],
                                "checks": [],
                            }
                        }
                    }
                }
            }
        },
    }
    out_dir = str(tmp_path)
    dest = tmp_path / "pdp_extension_schema.json"
    dest.write_text(json.dumps(existing))

    df = pd.DataFrame({"student_id": ["ABC"], "newcol": [5]})
    ext = generate_extension_schema(
        df,
        base_schema_file,
        "pdp",
        "student",
        extension_schema_path=None,
        output_dir=out_dir,
    )
    cols = ext["institutions"]["pdp"]["data_models"]["student"]["columns"]
    # should preserve original plus add newcol
    assert "orig" in cols and "newcol" in cols
    assert ext["version"] == "9.9.9"
