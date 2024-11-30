# ruff: noqa: F821
# mypy: ignore-errors

import pandera as pda
import pandera.typing as pt


class PDPLabeledDataSchema(pda.DataFrameModel):
    """
    Bare minimum columns required for a labeled dataset, i.e. a unique identifier
    and a target variable to be predicted; presumably, meany feature columns are
    also included, but *which* features will vary by school / model.
    """

    student_guid: pt.Series["string"]
    target: pt.Series["boolean"]

    class Config:
        coerce = True
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["student_guid"]
