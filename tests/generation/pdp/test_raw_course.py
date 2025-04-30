import faker
import pandas as pd
import pytest

from student_success_tool.dataio.schemas.pdp import RawPDPCourseDataSchema
from student_success_tool.generation.pdp import raw_course

FAKER = faker.Faker()
FAKER.add_provider(raw_course.Provider)


@pytest.mark.parametrize(
    ["normalize_col_names"],
    [(False,), (True,)],
)
def test_raw_course_record(normalize_col_names):
    obs = FAKER.raw_course_record(normalize_col_names=normalize_col_names)
    assert obs and isinstance(obs, dict)
    if normalize_col_names is True:
        df_obs = pd.DataFrame([obs])
        obs_valid = RawPDPCourseDataSchema.validate(df_obs, lazy=True)
        assert isinstance(obs_valid, pd.DataFrame)  # => data passed validation
