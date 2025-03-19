# ruff: noqa: F821
import typing as t
from functools import partial

import pandas as pd
import pandera as pda
import pandera.typing as pt
# from student_success_tool.analysis.pdp.schemas import RawPDPCohortDataSchema, RawPDPCourseDataSchema
from student_success_tool.schemas.pdp  import RawPDPCohortDataSchema, RawPDPCourseDataSchema

CompletedDevField = partial(
    pda.Field, nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
)

class RawPDPCourseDataSchema(RawPDPCourseDataSchema):
    # okay fine, no categories for hacc!
    # math_or_english_gateway: pt.Series[pd.CategoricalDtype] = pda.Field(
    #     nullable=True, dtype_kwargs={"categories": ["M", "E", "NA"]}
    # )
    math_or_english_gateway: pt.Series["string"] = pda.Field(
        nullable=True
    )
    term_program_of_study: t.Optional[pt.Series["string"]] = pda.Field(
        nullable=True
    )

    @pda.parser("math_or_english_gateway")
    def set_math_or_english_gateway_categories(cls, series):
        # somehow they've managed to append spaces to all of their values
        return series.str.strip()
        # # no, this doesn't work ... no idea why
        # # return series.cat.set_categories(["E", "M", "NA"], rename=True)
        # return (
        #     series.cat.add_categories(["E", "M", "NA"])
        #     .replace({"E  ": "E", "M  ": "M", "NA ": "NA"})
        #     .cat.remove_unused_categories()
        # )


class RawPDPCohortDataSchema(RawPDPCohortDataSchema):
    first_gen: pt.Series[pd.CategoricalDtype] = pda.Field(
        nullable=True, dtype_kwargs={"categories": ["Y", "N"]}
    )
    credential_type_sought_year_1: pt.Series[pd.CategoricalDtype] = pda.Field(
        dtype_kwargs={
            "categories": [
                "Less than 1-year certificate, less than Associates degree",
                "1-2 year certificate, less than Associates degree",
                "2-4 year certificate, less than Bachelor's degree",
                "Undergraduate Certificate or Diploma Program",
                "Associate's Degree",
                "Bachelor's Degree",
                "UNKNOWN",
            ]
        },
    )
    number_of_credits_attempted_year_1: pt.Series["Float32"] = pda.Field(
        nullable=True, ge=0.0
    )
    completed_dev_math_y_1: pt.Series[pd.CategoricalDtype] = CompletedDevField()
    completed_dev_english_y_1: pt.Series[pd.CategoricalDtype] = CompletedDevField()
    # added in 2025 data
    years_to_latest_associates_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    years_to_latest_certificate_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    years_to_latest_associates_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    years_to_latest_certificate_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    first_year_to_associates_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    first_year_to_certificate_at_cohort_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    first_year_to_associates_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )
    first_year_to_certificate_at_other_inst: t.Optional[pt.Series["Int8"]] = (
        pda.Field(nullable=True, ge=0, le=8)
    )