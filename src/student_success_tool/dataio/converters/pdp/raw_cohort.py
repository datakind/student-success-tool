import logging
import typing as t

import pandas as pd

LOGGER = logging.getLogger(__name__)


def rename_mangled_column_names(
    df: pd.DataFrame, *, overrides: t.Optional[dict[str, str]] = None
) -> pd.DataFrame:
    """
    Rename column names in ``df`` that have been mangled somewhere in the data generation
    and/or ingestion process, so they match the standard data schema.

    Args:
        df: Raw cohort dataset
        overrides: Mapping of raw (mangled) to raw (valid) column names in ``df``
            to be renamed. Note that repeatedly mangled columns are already included,
            so only specify this if the base set doesn't cover a school's situation.

    See Also:
        :class:`schemas.pdp.RawPDPCohortDataSchema`
    """
    rename_columns = {
        "attemptedgatewaymathyear_1": "attempted_gateway_math_year_1",
        "attemptedgatewayenglishyear_1": "attempted_gateway_english_year_1",
        "completedgatewaymathyear_1": "completed_gateway_math_year_1",
        "completedgatewayenglishyear_1": "completed_gateway_english_year_1",
        "gatewaymathgradey_1": "gateway_math_grade_y_1",
        "gatewayenglishgradey_1": "gateway_english_grade_y_1",
        "attempteddevmathy_1": "attempted_dev_math_y_1",
        "attempteddevenglishy_1": "attempted_dev_english_y_1",
        "completeddevmathy_1": "completed_dev_math_y_1",
        "completeddevenglishy_1": "completed_dev_english_y_1",
    }
    if overrides:
        rename_columns |= overrides
    # only rename mangled columns if they actually exist in the data
    rename_columns = {
        old_col: new_col
        for old_col, new_col in rename_columns.items()
        if old_col in df.columns
    }
    LOGGER.info("renaming mangled columns in raw cohort data: %s", rename_columns)
    return df.rename(columns=rename_columns)
