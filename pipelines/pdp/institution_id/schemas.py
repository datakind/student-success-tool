# ruff: noqa: F821
"""
School-specific overrides for school-agnostic data schemas, particularly for "raw" data.

References:
- https://pandera.readthedocs.io/en/stable/dataframe_models.html#schema-inheritance
- https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#basics-dtypes
"""

import pandas as pd  # noqa: F401
import pandera as pda  # noqa: F401
import pandera.typing as pt  # noqa: F401

from student_success_tool import schemas


class RawInstIDCohortDataSchema(schemas.pdp.RawPDPCohortDataSchema):
    pass


class RawInstIDCourseDataSchema(schemas.pdp.RawPDPCourseDataSchema):
    pass
