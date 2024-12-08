import faker
import pandas as pd
import pytest

from student_success_tool.analysis.pdp.schemas import RawPDPCohortDataSchema
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
