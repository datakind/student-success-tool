import faker
import pandas as pd
import pytest

from student_success_tool.dataio.schemas.pdp import RawPDPCohortDataSchema
from student_success_tool.generation.pdp import raw_cohort

FAKER = faker.Faker()
FAKER.add_provider(raw_cohort.Provider)


@pytest.mark.parametrize(
    ["min_cohort_yr", "max_cohort_yr", "normalize_col_names"],
    [(2010, None, False), (2015, 2020, True)],
)
def test_raw_cohort_record(min_cohort_yr, max_cohort_yr, normalize_col_names):
    obs = FAKER.raw_cohort_record(
        min_cohort_yr=min_cohort_yr,
        max_cohort_yr=max_cohort_yr,
        normalize_col_names=normalize_col_names,
    )
    assert obs and isinstance(obs, dict)
    if normalize_col_names is True:
        df_obs = pd.DataFrame([obs])
        obs_valid = RawPDPCohortDataSchema.validate(df_obs, lazy=True)
        assert isinstance(obs_valid, pd.DataFrame)  # => data passed validation
        print(df_obs)


def test_multiple_raw_cohort_records():
    institution_id = 123
    cohort_records = [
        FAKER.raw_cohort_record(normalize_col_names=True, institution_id=institution_id)
        for _ in range(10)
    ]
    df_cohort = pd.DataFrame(cohort_records)
    obs_valid = RawPDPCohortDataSchema.validate(df_cohort, lazy=True)
    assert isinstance(obs_valid, pd.DataFrame)  # => data passed validation
