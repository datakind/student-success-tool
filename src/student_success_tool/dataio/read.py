import logging
import pathlib
import typing as t

import pandas as pd
import pyspark.sql

try:
    import tomllib  # noqa
except ImportError:  # => PY3.10
    import tomli as tomllib  # noqa

LOGGER = logging.getLogger(__name__)


def from_csv_file(
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
        df = pd.read_csv(file_path, dtype="string", header="infer", **kwargs)  # type: ignore
    else:
        df = spark_session.read.csv(
            file_path,
            inferSchema=False,
            header=True,
            **kwargs,  # type: ignore
        ).toPandas()
    assert isinstance(df, pd.DataFrame)  # type guard
    LOGGER.info("loaded rows x cols = %s from '%s'", df.shape, file_path)
    return df


def from_delta_table(
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


def from_toml_file(file_path: str) -> dict[str, object]:
    """
    Read data from ``file_path`` and return it as a dict.

    Args:
        file_path: Path to file on disk from which data will be read.
    """
    fpath = pathlib.Path(file_path).resolve()
    with fpath.open(mode="rb") as f:
        data = tomllib.load(f)
    LOGGER.info("loaded config from '%s'", fpath)
    assert isinstance(data, dict)  # type guard
    return data
