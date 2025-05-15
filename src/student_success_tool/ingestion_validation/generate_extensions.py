import sys
import os
import json
from typing import Union, List, Any

import pandas as pd

# import your validator (assumes validate_dataset and HardValidationError live in validate_dataset.py)
from .validation import validate_dataset, normalize_col, HardValidationError


def load_json(path: str) -> Any:
    """Load JSON from a file, returning {} on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def infer_column_schema(series: pd.Series, cate_threshold: int = 10) -> Any:
    """
    Infer a minimal Pandera‐style schema for a pandas Series.
    """
    import numpy as np

    non_null = series.dropna()
    uniques = non_null.unique().tolist()
    # small categorical
    if 1 < len(uniques) <= cate_threshold:
        cats = [(v.item() if isinstance(v, np.generic) else v) for v in uniques]
        return {
            "dtype": "category",
            "categories": cats,
            "coerce": True,
            "nullable": bool(series.isna().any()),
            "required": True,
            "aliases": [],
            "checks": [{"type": "isin", "args": [cats]}],
        }

    # numeric / datetime / bool / string fallback
    dt = series.dtype
    if pd.api.types.is_integer_dtype(dt):
        dtype = "float64"
        checks = [{"type": "ge", "args": [0]}]
    elif pd.api.types.is_float_dtype(dt):
        dtype = "float64"
        checks = [{"type": "ge", "args": [0.0]}]
    elif pd.api.types.is_bool_dtype(dt):
        dtype = "boolean"
        checks = []
    elif pd.api.types.is_datetime64_any_dtype(dt):
        dtype = "datetime64[ns]"
        checks = [{"type": "not_null", "args": []}]
    else:
        dtype = "string"
        checks = []

    return {
        "dtype": dtype,
        "coerce": True,
        "nullable": bool(series.isna().any()),
        "required": True,
        "aliases": [],
        "checks": checks,
    }


def generate_extension_schema(
    df: Union[pd.DataFrame, str], models: Union[str, List[str]], institution_id: str
) -> Any:
    """
    1) run validate_dataset(...) to detect hard errors or extra_columns
    2) infer schema for each extra column (for extension)
    3) load or init the institution extension JSON
    4) write inferred specs under each model's .data_models[model].columns
    """
    # load & normalize DataFrame
    if isinstance(df, str):
        df = pd.read_csv(df)
    df = df.rename(columns=lambda c: normalize_col(c))

    # run validation; catch HardValidationError to extract extras
    base_schema_path = "'/Volumes/staging_sst_01/default/schema/base_schema.json'"
    try:
        # no hard errors (i.e. no missing_required, extra_columns, or schema errors)
        _ = validate_dataset(df, models, institution_id)
        extras: List[str] = []
    except HardValidationError as e:
        # if missing_required or schema_errors, cannot proceed
        if e.missing_required or e.schema_errors:
            print("Validation FAILED:")
            print(str(e))
            sys.exit(1)
        # otherwise, treat extra_columns as the ones to extend
        extras = e.extra_columns or []

    if not extras:
        print(f"No extra columns to extend for models={models!r}")
        return {}

    # keep only extras actually in df
    extras = [c for c in extras if c in df.columns]
    print(f"Will infer {len(extras)} extra columns: {extras}")

    # infer a spec for each extra column
    inferred = {col: infer_column_schema(df[col]) for col in extras}

    # prepare extension JSON
    output_dir = f"/Volumes/staging_sst_01/{institution_id}_bronze/schema/"
    os.makedirs(output_dir, exist_ok=True)
    inst_fn = f"{institution_id}_schema_extension.json"
    out_path = os.path.join(output_dir, inst_fn)

    extension = load_json(out_path)
    if not extension:
        base_version = load_json(base_schema_path).get("version", "1.0.0")
        extension = {
            "version": base_version,
            "institutions": {institution_id: {"data_models": {}}},
        }

    inst_block = (
        extension.setdefault("institutions", {})
        .setdefault(institution_id, {})
        .setdefault("data_models", {})
    )

    # ensure models is a list
    if isinstance(models, str):
        model_list = [models]
    else:
        model_list = models

    # insert inferred extras under each requested model
    for model in model_list:
        cols_block = inst_block.setdefault(model, {}).setdefault("columns", {})
        cols_block.update(inferred)

    # write back
    with open(out_path, "w") as f:
        json.dump(extension, f, indent=2)

    print(f"Wrote/updated extension schema to {out_path}")
    return extension
