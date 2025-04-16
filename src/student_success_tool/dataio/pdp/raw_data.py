import functools as ft
import logging
import typing as t

import pandas as pd
import pyspark.sql

from ... import utils
from .. import read

try:
    import pandera as pda
except ModuleNotFoundError:
    from ... import utils

    utils.databricks.mock_pandera()

    import pandera as pda

LOGGER = logging.getLogger(__name__)


def read_raw_course_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
    dttm_format: str = "%Y%m%d",
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read raw PDP course data from table (in Unity Catalog) or file (in CSV format),
    and parse+validate it via ``schema`` .

    Args:
        table_path
        file_path
        schema: "DataFrameModel", such as those specified in :mod:`schemas` ,
            used to parse and validate the raw data. If None, parsing/validation
            is skipped, and the raw data is returned as-is.
        dttm_format: Datetime format for "Course Begin/End Date" columns.
        converter_func: If the raw data is incompatible with ``schema`` ,
            provide a function that takes the raw dataframe as its sole input,
            performs whatever (minimal) transformations necessary to bring the data
            into line with ``schema`` , and then returns it. This converted dataset
            will then be passed into ``schema`` , if specified.
            NOTE: Allowances for minor differences in the data should be implemented
            on the school-specific schema class directly. This function is intended
            to handle bigger problems, such as duplicate ids or borked columns.
        spark_session: Required if reading data from ``table_path`` , and optional
            if reading data from ``file_path`` .
        **kwargs: Additional arguments passed as-is into underlying read func.
            Note that raw data is always read in as "string" dtype, then coerced
            into the correct dtypes using ``schema`` .

    See Also:
        - :func:`read_data_from_csv_file()`
        - :func:`read_data_from_delta_table()`

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.dataframe.model.DataFrameModel.html#pandera.api.dataframe.model.DataFrameModel.validate
    """
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    elif table_path is not None and spark_session is None:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        read.from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else read.from_delta_table(table_path, spark_session)  # type: ignore
    )
    # apply to the data what pandera calls "parsers" before validation
    # ideally, all these operations would be dataframe parsers on the schema itself
    # but pandera applies core before custom parsers under the hood :/
    df = (
        # standardize column names
        df.rename(columns=utils.misc.convert_to_snake_case)
        # standardize certain column values
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in ("academic_term",)
            }
            # help pandas to parse non-standard datetimes... read_csv() struggles
            | {
                col: ft.partial(_parse_dttm_values, col=col, fmt=dttm_format)
                for col in ("course_begin_date", "course_end_date")
            }
        )
    )
    return _maybe_convert_maybe_validate_data(df, converter_func, schema)


def read_raw_cohort_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read raw PDP cohort data from table (in Unity Catalog) or file (in CSV format),
    and parse+validate it via ``schema`` .

    Args:
        table_path
        file_path
        schema: "DataFrameModel", such as those specified in :mod:`schemas` ,
            used to parse and validate the raw data. If None, parsing/validation
            is skipped, and the raw data is returned as-is.
        converter_func: If the raw data is incompatible with ``schema`` ,
            provide a function that takes the raw dataframe as its sole input,
            performs whatever (minimal) transformations necessary to bring the data
            into line with ``schema`` , and then returns it. This converted dataset
            will then be passed into ``schema`` , if specified.
            NOTE: Allowances for minor differences in the data should be implemented
            on the school-specific schema class directly. This function is intended
            to handle bigger problems, such as duplicate ids or borked columns.
        spark_session: Required if reading data from ``table_path`` , and optional
            if reading data from ``file_path`` .
        **kwargs: Additional arguments passed as-is into underlying read func.
            Note that raw data is always read in as "string" dtype, then coerced
            into the correct dtypes using ``schema`` .

    See Also:
        - :func:`read_data_from_csv_file()`
        - :func:`read_data_from_delta_table()`

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.dataframe.model.DataFrameModel.html#pandera.api.dataframe.model.DataFrameModel.validate
    """
    if not bool(table_path) ^ bool(file_path):
        raise ValueError("exactly one of table_path or file_path must be specified")
    elif table_path is not None and spark_session is None:
        raise ValueError("spark session must be given when reading data from table")

    df = (
        read.from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else read.from_delta_table(table_path, spark_session)  # type: ignore
    )
    # apply to the data what pandera calls "parsers" before validation
    # ideally, all these operations would be dataframe parsers on the schema itself
    # but pandera applies core before custom parsers under the hood :/
    df = (
        # standardize column names
        df.rename(columns=utils.misc.convert_to_snake_case)
        # standardize column values
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            # for practical reasons, this is the only place where it's easy to do so
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in ("cohort_term",)
            }
            # replace "UK" with null in GPA cols, so we can coerce to float via schema
            | {
                col: ft.partial(_replace_values_with_null, col=col, to_replace="UK")
                for col in ("gpa_group_term_1", "gpa_group_year_1")
            }
            # help pandas to coerce string "1"/"0" values into True/False
            | {
                col: ft.partial(_cast_to_bool_via_int, col=col)
                for col in ("retention", "persistence")
            }
        )
    )
    return _maybe_convert_maybe_validate_data(df, converter_func, schema)


def _maybe_convert_maybe_validate_data(
    df: pd.DataFrame,
    converter_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    schema: t.Optional[type[pda.DataFrameModel]] = None,
) -> pd.DataFrame:
    # HACK: we're hiding this pandera import here so databricks doesn't know about it
    # pandera v0.23+ pulls in pandas v2.1+ while databricks runtimes are stuck in v1.5
    # resulting in super dumb dependency errors when loading automl trained models
    import pandera.errors

    if converter_func is not None:
        LOGGER.info("applying %s converter to raw data", converter_func)
        df = converter_func(df)
    if schema is None:
        return df
    else:
        try:
            df_validated = schema.validate(df, lazy=True)
            assert isinstance(df_validated, pd.DataFrame)
            return df_validated
        except pandera.errors.SchemaErrors:
            LOGGER.error("unable to parse/validate raw data")
            raise


def _parse_dttm_values(df: pd.DataFrame, *, col: str, fmt: str) -> pd.Series:
    return pd.to_datetime(df[col], format=fmt)


def _uppercase_string_values(df: pd.DataFrame, *, col: str) -> pd.Series:
    return df[col].str.upper()


def _replace_values_with_null(
    df: pd.DataFrame, *, col: str, to_replace: str | list[str]
) -> pd.Series:
    return df[col].replace(to_replace=to_replace, value=None)


def _cast_to_bool_via_int(df: pd.DataFrame, *, col: str) -> pd.Series:
    return (
        df[col]
        .astype("string")
        .map(
            {
                "1": True,
                "0": False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
            }
        )
        .astype("boolean")
    )
