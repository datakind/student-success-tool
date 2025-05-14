## Overview

SST tools for validating and extending data schemas using Pandera. It consists of two main components:

1. **`validation.py`**: Validates incoming datasets against a base schema (and any institution-specific extensions).
2. **`generate_schema.py`**: Infers and generates institution extension schemas for any dataset columns not covered by the base schema.

### Directory Structure

```
├── base_schema.json         # The core, organization-wide schema definition
├── validation.py            # Validation engine (hard- and soft-errors) using Pandera
├── generate_schema.py       # Schema extension generator for custom institution fields
├── extensions/              # Output folder where generated extension schemas are saved
│   └── <institution>_extension_schema.json
└── README.md                # This file
```

#### Base Schema (`base_schema.json`)

* Contains the base schema definitions for each data model, from https://docs.google.com/spreadsheets/d/1zOLv2VOIhDpy6f_2KdOJqLOgA9GNhxW8ZUwneMPF-8A/edit?gid=1337889658#gid=1337889658.
* Defines required and optional columns, data types, and validation checks.

#### Validation (`validation.py`)

This script performs full dataset validation:

1. **Load & Normalize**: Reads a CSV or DataFrame, normalizes column names (only lowercase and underscores).
2. **Schema Merge**: Loads `base_schema.json` and any institution-specific extension from `extensions/`.
3. **Column Discovery**:

   * Flags **extra** columns not in the merged schema (hard error).
   * Flags **missing required** columns (hard error).
   * Flags **missing optional** columns (soft error).
4. **Pandera Validation**: Builds a `DataFrameSchema` from merged specs and runs element-wise checks.

   * **Hard errors** (schema mismatch or check failures) raise a `HardValidationError`.
   * **Soft errors** (missing optional) are reported in the returned status.

**Usage**:

```bash
from validation import validate_dataset
ingestion_data = pd.read_csv('data.csv')
validate_dataset(ingestion_data, 'base_schema.json', [student | semester | course], 'institution_id', "/extensions/inst_id_extension_schema.json")
```

**example**

```bash
from validation import validate_dataset
ingestion_data = pd.read_csv('nscc_student.csv')
validate_dataset(
      df=ingestion_data, 
      base_schema_path='base_schema.json', 
      models=[student], 
      institution='nscc', 
      extension_schema_path="/extensions/nscc_extension_schema.json")
```

The script returns a JSON-like dict:

```python
{
  "validation_status": "passed"  # or "passed_with_soft_errors"
  "missing_optional": [ ... ]
}
```

#### Exceptions

* `HardValidationError`: Raised if there are missing required columns, unexpected columns, or Pandera schema errors.

#### Schema Extension Generation (`generate_schema.py`)

This tool helps extend the base schema by inferring specs for columns not covered.

1. **Validate Dataset**: Calls `validate_dataset(...)` to identify `extra_columns` in the input.
2. **Infer Specs**: For each extra column, inspects data types and values to build a minimal Pandera‐style spec.
3. **Load or Initialize** the institution’s extension JSON under `extensions/`.
4. **Update Extension**: Inserts inferred specs into the institution’s `data_models` & writes back.

**Usage**:
```bash
from generate_extensions import generate_extension_schema
generate_extension_schema(
    df=base_data_from_institution.csv,
    base_schema_path='base_schema.json',
    institution='institution_id',
    models=[student or semester or course],
    output_dir='extensions'
)
```

**example**
```bash
from generate_extensions import generate_extension_schema
generate_extension_schema(
    df='nscc.csv',
    base_schema_path='base_schema.json',
    institution='nscc',
    models=[student],
    output_dir='extensions'
)
```

* If no extra columns are found, the tool exits without modifying files.
* Generated extension schemas are saved as `<output-dir>/<institution>_extension_schema.json`.

#### Extensions Folder

* Contains JSON files for institution-specific schema extensions.
* Example: `extensions/pdp_extension_schema.json`.
* Each extension JSON mirrors the base schema structure, adding only the extra columns for requested models.
