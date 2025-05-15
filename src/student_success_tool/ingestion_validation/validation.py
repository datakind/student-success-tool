import json
import os
import re
from typing import Union, List, Dict, Any

import pandas as pd
from pandera import Column, Check, DataFrameSchema
from pandera.errors import SchemaErrors


class HardValidationError(Exception):
    def __init__(
        self,
        missing_required: Any = None,
        extra_columns: Any = None,
        schema_errors: Any = None,
        failure_cases: Any = None,
    ):
        self.missing_required = missing_required or []
        self.extra_columns = extra_columns or []
        self.schema_errors = schema_errors
        self.failure_cases = failure_cases
        parts = []
        if self.missing_required:
            parts.append(f"Missing required columns: {self.missing_required}")
        if self.extra_columns:
            parts.append(f"Unexpected columns: {self.extra_columns}")
        if self.schema_errors is not None:
            parts.append(f"Schema errors: {self.schema_errors}")
        super().__init__("; ".join(parts))


def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_json(path: str) -> Any:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load JSON schema at {path}: {e}")


def merge_model_columns(
    base_schema: dict, extension_schema: Any, institution: str, model: str
) -> dict:
    base_models = base_schema.get("base", {}).get("data_models", {})
    if model not in base_models:
        raise KeyError(f"Model '{model}' not in base schema")
    merged = dict(base_models[model].get("columns", {}))
    if extension_schema:
        inst_block = extension_schema.get("institutions", {}).get(institution, {})
        ext_models = inst_block.get("data_models", {})
        if model in ext_models:
            merged.update(ext_models[model].get("columns", {}))
    return merged


def build_schema(specs: Dict[str, dict]) -> DataFrameSchema:
    columns = {}
    for canon, spec in specs.items():
        names = [canon] + spec.get("aliases", [])
        pattern = r"^(?:" + "|".join(map(re.escape, names)) + r")$"
        checks = []
        for chk in spec.get("checks", []):
            factory = getattr(Check, chk["type"])
            checks.append(factory(*chk.get("args", []), **chk.get("kwargs", {})))

        columns[pattern] = Column(
            name=pattern,
            regex=True,
            dtype=spec["dtype"],
            nullable=spec["nullable"],
            required=spec.get("required", False),
            checks=checks or None,
            coerce=spec.get("coerce", False),
        )
    return DataFrameSchema(columns, strict=False)


def validate_dataset(
    df: Union[pd.DataFrame, str],
    models: Union[str, List[str]],
    institution_id: str,
) -> Dict[str, Any]:
    if isinstance(df, str):
        df = pd.read_csv(df)
    df = df.rename(columns={c: normalize_col(c) for c in df.columns})
    incoming = set(df.columns)

    # 1) load schemas
    base_schema_path = "/Volumes/staging_sst_01/default/schema/base_schema.json"
    base_schema = load_json(base_schema_path)
    ext_schema = None
    extension_schema_path = f"/Volumes/staging_sst_01/{institution_id}_bronze/schema/{institution_id}_schema_extension.json"
    if extension_schema_path and os.path.exists(extension_schema_path):
        ext_schema = load_json(extension_schema_path)

    # 2) merge requested models
    if isinstance(models, str):
        model_list = [models]
    else:
        model_list = models

    merged_specs: Dict[str, dict] = {}
    for m in model_list:
        specs = merge_model_columns(base_schema, ext_schema, institution_id, m)
        merged_specs.update(specs)

    # 3) build canon → set(normalized names)
    canon_to_norms: Dict[str, set] = {
        canon: {normalize_col(alias) for alias in [canon] + spec.get("aliases", [])}
        for canon, spec in merged_specs.items()
    }

    # 4) find extra / missing
    all_norms = set().union(*canon_to_norms.values()) if canon_to_norms else set()
    extra_columns = sorted(incoming - all_norms)

    missing_required = [
        canon
        for canon, norms in canon_to_norms.items()
        if merged_specs[canon].get("required", False) and norms.isdisjoint(incoming)
    ]

    missing_optional = [
        canon
        for canon, norms in canon_to_norms.items()
        if not merged_specs[canon].get("required", False) and norms.isdisjoint(incoming)
    ]

    # Hard-fail on missing required or any extra columns
    if missing_required or extra_columns:
        raise HardValidationError(
            missing_required=missing_required, extra_columns=extra_columns
        )

    # 5) build Pandera schema & validate (hard-fail on any error)
    schema = build_schema(merged_specs)
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as err:
        raise HardValidationError(
            schema_errors=err.schema_errors,
            failure_cases=err.failure_cases.to_dict(orient="records"),
        )

    print(
        {
            "validation_status": (
                "passed_with_soft_errors" if missing_optional else "passed"
            ),
            "missing_optional": missing_optional,
        }
    )
    # 6) success (with possible soft misses)
    return {
        "validation_status": (
            "passed_with_soft_errors" if missing_optional else "passed"
        ),
        "missing_optional": missing_optional,
    }
