import functools as ft
import logging
import time
import typing as t

import pandas as pd
import pandera as pda
import pandera.errors
import pyspark.sql

from . import utils

LOGGER = logging.getLogger(__name__)


def read_raw_pdp_course_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[pda.DataFrameModel] = None,
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
        read_data_from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else read_data_from_delta_table(table_path, spark_session)  # type: ignore
    )
    df = (
        # make minimal changes before parsing/validating via schema
        # standardize column names, for convenience/consistency
        df.rename(columns=utils.convert_to_snake_case)
        # basically, any operations needed for dtype coercion to work correctly
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in ("academic_term", "student_age", "race")
            }
            # help pandas to parse non-standard datetimes... read_csv() struggles
            | {
                col: ft.partial(_parse_dttm_values, col=col, fmt=dttm_format)
                for col in ("course_begin_date", "course_end_date")
            }
        )
    )
    return _maybe_convert_maybe_validate_data(df, converter_func, schema)


def read_raw_pdp_cohort_data(
    *,
    table_path: t.Optional[str] = None,
    file_path: t.Optional[str] = None,
    schema: t.Optional[pda.DataFrameModel] = None,
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
        read_data_from_csv_file(file_path, spark_session, **kwargs)  # type: ignore
        if file_path
        else read_data_from_delta_table(table_path, spark_session)  # type: ignore
    )
    df = (
        # make minimal changes before parsing/validating via schema
        # standardize column names, for convenience/consistency
        df.rename(columns=utils.convert_to_snake_case)
        # basically, any operations needed for dtype coercion to work correctly
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            # for practical reasons, this is the only place where it's easy to do so
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in (
                    "cohort_term",
                    "enrollment_type",
                    "enrollment_intensity_first_term",
                    "student_age",
                    "race",
                    "most_recent_bachelors_at_other_institution_locale",
                    "most_recent_associates_or_certificate_at_other_institution_locale",
                    "most_recent_last_enrollment_at_other_institution_locale",
                    "first_bachelors_at_other_institution_locale",
                    "first_associates_or_certificate_at_other_institution_locale",
                )
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
    schema: t.Optional[pda.DataFrameModel] = None,
) -> pd.DataFrame:
    if converter_func is not None:
        LOGGER.info("applying %s converter to raw data", converter_func)
        df = converter_func(df)
    if schema is None:
        return df
    else:
        try:
            df = schema.validate(df, lazy=True)  # type: ignore
            return df  # type: ignore
        except pandera.errors.SchemaErrors:
            LOGGER.error("unable to parse/validate raw data")
            raise


def read_raw_pdp_course_data_from_file(
    fpath: str,
    *,
    schema: t.Optional[pda.DataFrameModel],
    dttm_format: str = "%Y%m%d.0",
    preprocess_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read a raw PDP course data file stored on disk at ``fpath`` in CSV format,
    and parse+validate it via ``schema`` .

    Args:
        file_path
        schema: "DataFrameModel", such as those specified in :mod:`pdp.schemas` ,
            used to parse and validate the raw data. If None, parsing/validation
            is skipped, and the raw data is returned as-is.
        dttm_format: Datetime format for "Course Begin/End Date" columns.
        preprocess_func: If the raw data at ``fpath`` is incompatible with ``schema`` ,
            provide a function that takes the raw dataframe as its sole input,
            performs whatever (minimal) transformations necessary to bring the data
            into line with ``schema`` , and then returns it. This preprocessed dataset
            will then be passed into ``schema`` , if specified.
            NOTE: Allowances for minor differences in the data should be implemented
            on the school-specific schema class directly. This function is intended
            to handle bigger problems, such as dupe ids or borked columns.
        **kwargs: Additional arguments passed as-is into :func:`pd.read_csv()` .
            Note that ``dtype`` is hard-coded, so you can't specify it again here.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/course-level-analysis-ready-file-data-dictionary
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.dataframe.model.DataFrameModel.html#pandera.api.dataframe.model.DataFrameModel.validate
    """
    df = (
        # read everything as a string to start, for maximum flexibility
        pd.read_csv(fpath, dtype="string", **kwargs)  # type: ignore
        # make minimal changes before parsing/validating via schema
        # standardize column names, for convenience/consistency
        .rename(columns=utils.convert_to_snake_case)
        # basically, any operations needed for dtype coercion to work correctly
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in ("academic_term", "student_age", "race")
            }
            # help pandas to parse non-standard datetimes... read_csv() struggles
            | {
                col: ft.partial(_parse_dttm_values, col=col, fmt=dttm_format)
                for col in ("course_begin_date", "course_end_date")
            }
        )
    )
    LOGGER.info("loaded rows x cols = %s of course data from '%s'", df.shape, fpath)
    assert isinstance(df, pd.DataFrame)  # type guard
    if preprocess_func is not None:
        LOGGER.info("applying %s preprocessor to raw dataset", preprocess_func)
        df = preprocess_func(df)
    if schema is None:
        return df
    else:
        try:
            df = schema.validate(df, lazy=True)
            return df  # type: ignore
        except pandera.errors.SchemaErrors:
            LOGGER.error("unable to parse/validate course data")
            raise


def read_raw_pdp_cohort_data_from_file(
    fpath: str,
    *,
    schema: t.Optional[pda.DataFrameModel],
    preprocess_func: t.Optional[t.Callable[[pd.DataFrame], pd.DataFrame]] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read a raw PDP cohort data file stored on disk at ``fpath`` in CSV format.

    Args:
        fpath
        schema: "DataFrameModel", such as those specified in :mod:`pdp.schemas` ,
            used to parse and validate the raw data. If None, parsing/validation
            is skipped, and the raw data is returned as-is.
        preprocess_func: If the raw data at ``fpath`` is incompatible with ``schema`` ,
            provide a function that takes the raw dataframe as its sole input,
            performs whatever (minimal) transformations necessary to bring the data
            into line with ``schema`` , and then returns it. This preprocessed dataset
            will then be passed into ``schema`` , if specified.
            NOTE: Allowances for minor differences in the data should be implemented
            on the school-specific schema class directly. This function is intended
            to handle bigger problems, such as dupe ids or borked columns.
        **kwargs: Additional arguments passed as-is into :func:`pd.read_csv()` .
            Note that ``dtype`` is hard-coded to "string", so you can't specify it here.

    References:
        - https://help.studentclearinghouse.org/pdp/knowledge-base/cohort-level-analysis-ready-file-data-dictionary
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.dataframe.model.DataFrameModel.html#pandera.api.dataframe.model.DataFrameModel.validate
    """
    df = (
        # read everything as a string to start, for maximum flexibility
        pd.read_csv(fpath, dtype="string", **kwargs)  # type: ignore
        # make minimal changes before parsing/validating via schema
        # standardize column names, for convenience/consistency
        .rename(columns=utils.convert_to_snake_case)
        # basically, any operations needed for dtype coercion to work correctly
        .assign(
            # uppercase string values for some cols to avoid case inconsistency later on
            # for practical reasons, this is the only place where it's easy to do so
            **{
                col: ft.partial(_uppercase_string_values, col=col)
                for col in (
                    "cohort_term",
                    "enrollment_type",
                    "enrollment_intensity_first_term",
                    "student_age",
                    "race",
                    "most_recent_bachelors_at_other_institution_locale",
                    "most_recent_associates_or_certificate_at_other_institution_locale",
                    "most_recent_last_enrollment_at_other_institution_locale",
                    "first_bachelors_at_other_institution_locale",
                    "first_associates_or_certificate_at_other_institution_locale",
                )
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
    LOGGER.info("loaded rows x cols = %s of cohort data from '%s'", df.shape, fpath)
    assert isinstance(df, pd.DataFrame)  # type guard
    if preprocess_func is not None:
        LOGGER.info("applying %s preprocessor to raw dataset", preprocess_func)
        df = preprocess_func(df)
    if schema is None:
        return df
    else:
        try:
            df = schema.validate(df, lazy=True)
            return df  # type: ignore
        except pandera.errors.SchemaErrors:
            LOGGER.error("unable to parse/validate cohort data")
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


def read_data_from_csv_file(
    file_path: str,
    spark_session: t.Optional[pyspark.sql.SparkSession] = None,
    **kwargs: object,
) -> pd.DataFrame:
    """
    Read data from a CSV file at ``file_path`` and return it as a DataFrame.

    Args:
        file_path: Path to file on disk from which data will be read.
        spark_session: If given, data is loaded via ``pyspark.sql.DataFrameReader.csv`` ;
            otherwise, data is loaded via :func:`pandas.read_csv()` .
        **kwargs: Additional arguments passed as-is into underlying read func.

    See Also:
        - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.csv.html
    """
    if spark_session is None:
        df = pd.read_csv(file_path, dtype="string", **kwargs)  # type: ignore
    else:
        df = spark_session.read.csv(
            file_path, inferSchema=False, header=True, **kwargs
        ).toPandas()  # type: ignore
    assert isinstance(df, pd.DataFrame)  # type guard
    LOGGER.info("loaded rows x cols = %s from '%s'", df.shape, file_path)
    return df


def read_data_from_delta_table(
    table_path: str, spark_session: pyspark.sql.SparkSession
) -> pd.DataFrame:
    """
    Read data from a table in Databricks Unity Catalog and return it as a DataFrame.

    Args:
        table_path: Path in Unity Catalog from which data will be read,
            including the full three-level namespace: ``catalog.schema.table`` .
        spark_session: Entry point to using spark dataframes and the databricks integration.

    See Also:
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrameReader.html#pyspark.sql.DataFrameReader
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/spark_session.html
    """
    df = spark_session.read.format("delta").table(table_path).toPandas()
    assert isinstance(df, pd.DataFrame)  # type guard
    LOGGER.info("loaded rows x cols = %s of data from '%s'", df.shape, table_path)
    return df


def write_data_to_delta_table(
    df: pd.DataFrame, table_path: str, spark_session: pyspark.sql.SparkSession
) -> None:
    """
    Write pandas DataFrame to Databricks Unity Catalog.

    Args:
        df
        table_path: Path in Unity Catalog to which ``df`` will be written,
            including the full three-level namespace: ``catalog.schema.table`` .
        spark_session: Entry point to using spark dataframes and the databricks integration.

    See Also:
        - https://docs.databricks.com/en/delta/drop-table.html#when-to-replace-a-table
        - https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/spark_session.html
    """
    start_time = time.time()
    LOGGER.info("saving data to %s ...", table_path)
    df_spark = spark_session.createDataFrame(
        df.rename(columns=utils.convert_to_snake_case)
    )
    (
        pyspark.sql.DataFrameWriterV2(df_spark, table_path)
        .options(format="delta")
        # this *should* do what databricks recomends -- and retains table history!
        .createOrReplace()
    )
    run_time = time.time() - start_time

    table_rows = spark_session.sql(f"SELECT COUNT(*) FROM {table_path}").collect()
    if table_rows[0][0] != len(df):
        raise IOError(
            f"{table_rows[0][0]} written to delta table, "
            f"but {len(df)} rows in original dataframe"
        )

    history = spark_session.sql(f"DESCRIBE history {table_path} LIMIT 1").collect()
    verno = int(history[0][0])
    LOGGER.info("data saved to %s (v%s) in %s seconds", table_path, verno, run_time)
