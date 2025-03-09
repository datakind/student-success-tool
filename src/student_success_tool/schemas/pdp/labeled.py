# ruff: noqa: F821
# mypy: ignore-errors
import typing as t

import pandas as pd

try:
    import pandera as pda
    import pandera.typing as pt
except ModuleNotFoundError:
    from ... import utils

    utils.databricks.mock_pandera()

    import pandera as pda
    import pandera.typing as pt


class PDPLabeledDataSchema(pda.DataFrameModel):
    """
    Bare minimum columns required for a labeled dataset, i.e. a unique identifier
    and a target variable to be predicted; presumably, meany feature columns are
    also included, but *which* features will vary by school / model.
    """

    student_id: pt.Series["string"]
    target: pt.Series["boolean"]
    split: t.Optional[pt.Series[pd.CategoricalDtype]] = pda.Field(
        dtype_kwargs={"categories": ["train", "test", "validate"]}
    )

    class Config:
        coerce = True
        unique_column_names = True
        add_missing_columns = False
        drop_invalid_rows = False
        unique = ["student_id"]
