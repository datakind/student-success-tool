import warnings

import numpy as np
import pandas as pd
import pytest

from student_success_tool.modeling.inference import (
    select_top_features_for_display,
)

# Ignore warnings that might clutter the output
warnings.simplefilter(action="ignore")


# Set up the common data as fixtures in pytest
@pytest.fixture(scope="module")
def predicted_probabilities():
    return [0.96375323, 0.33946405, 0.07414102]


@pytest.fixture(scope="module")
def test_data():
    return {
        "UNIQUE_ID": [1, 2, 3],
        "FULL_TIME_ONLY_TERMS": [0, 0, 0],
        "PART_TIME_ONLY_TERMS": [0, 0, 0],
        "BOTH_FULL_PART_TIME_TERMS": [1, 1, 1],
        "FRACTION_FULL_TIME_TERMS": [0.67, 0.71, 0.6],
        "TOTAL_FAILED_HOURS": [49.0, 4.5, 0.0],
        "TOTAL_EXCLUDED_HOURS": [39.0, 16.5, 0.0],
        "TOTAL_HOURS_ENROLLED": [61.0, 78.5, 47.0],
        "TOTAL_PASSED_HOURS": [12.0, 74.0, 47.0],
        "CREDITS_HOURS_SEMESTER_COMPLETED_PERF_SUM": [47.0, 74.0, 47.0],
        "AVG_CREDITS_SEM_COMPLETED": [7.83, 10.6, 9.4],
        "STDEV_CREDITS_SEM": [4.67, 4.43, 5.50],
        "COURSE_PASS_RATE": [0.19, 0.94, 1.0],
        "CREDITS_ENROLLED_LAST_SEM": [7.0, 14.0, 9.0],
        "CREDITS_ENROLLED_LAST_2_SEM": [19.0, 21.0, 22.0],
        "CREDITS_ENROLLED_LAST_3_SEM": [30.0, 35.0, 34.0],
        "CREDITS_ENROLLED_LAST_4_SEM": [35.0, 50.0, 47.0],
        "HAS_ANY_FINANCIAL_AID": [1, 0, 1],
        "IR_TOTAL_FA_AWD_AMT_SUM": [0.0, 0.0, 10106.5],
        "IR_TOTAL_GRANTS_NB_AWD_AMT_SUM": [0.0, 0.0, 7291.5],
        "IR_STUDENT_LOANS_NB_AWD_AMT_SUM": [0.0, 0.0, 0.0],
        "IR_FED_WORK_STUDY_NB_AWD_AMT_SUM": [0.0, 0.0, 0.0],
        "PELL_AMOUNT_AWARD_SUM": [2155.0, 0.0, 797.5],
        "TAP_AMOUNT_AWARD_SUM": [635.0, 0.0, 1019.5],
        "PELL_FLAG_COUNT": [1, 0, 2],
        "TAP_FLAG_COUNT": [2, 0, 2],
        "PELL_TO_TAP_AMOUNT_RATIO": [3.40, 2.41, 0.78],
        "PELL_TO_TAP_COUNT_RATIO": [0.5, 1.31, 1.0],
        "GPA_SEMESTER_AVERAGE": [0.96, 3.23, 3.94],
        "GPA_SEMESTER_MIN": [0.0, 2.56, 3.83],
        "GPA_SEMESTER_STDEV": [1.69, 0.37, 0.0804],
        "GPA_LAST_SEM": [0.0, 3.15, 4.0],
        "GPA_LAST_2_SEM": [0.0, 2.85, 4.0],
        "GPA_LAST_3_SEM": [0.0, 2.97, 3.94],
        "GPA_LAST_4_SEM": [0.0, 3.07, 3.94],
        "YEARS_ENROLLED": [2.42, 2.25, 1.67],
        "MAJOR_CHANGES_NUM": [2.0, 0.0, 0.0],
        "HAS_ASSOCIATE_DEGREE": [0, 1, 1],
        "IS_JUSTICE_ACADEMY": [0, 1, 1],
    }


@pytest.fixture(scope="module")
def test_df(test_data):
    return pd.DataFrame(test_data)


@pytest.fixture(scope="module")
def shap_values():
    return np.array(
        [
            [
                0.011118,
                0.0,
                0.0907295,  # highest
                0.00252756,
                0.00398351,
                0.01459706,
                -0.00812746,
                0.00510075,
                -0.00110377,
                0.0,
                0.01422942,
                0.03684777,
                0.0345322,
                0.02334324,
                0.05655583,  # second
                0.0286916,
                0.01193176,
                -0.01909896,
                0.03867024,
                0.0,
                0.0,
                0.01072554,
                0.00422347,
                0.05371293,  # third
                0.01439325,
                0.0,
                -0.00218049,
                -0.00182518,
                0.0,
                0.0,
                0.00295084,
                0.00025675,
                -0.00991723,
                0.02664684,
                0.01449639,
                -0.00168721,
                0.01817514,
                0.03013451,
            ],
            [
                -0.00854077,
                0.0,
                -0.04005222,
                0.0,
                -0.00274858,
                -0.00887971,
                0.00842682,
                -0.00609056,
                0.0,
                0.0,
                -0.00404544,
                -0.01132483,
                -0.0113697,
                -0.00791739,
                -0.01783329,
                -0.00582056,
                -0.00432695,
                0.03870561,
                -0.02091931,
                0.0,
                0.0,
                0.00922927,
                0.00562887,
                -0.02678826,
                -0.00872366,
                0.0,
                0.0,
                -0.01235557,
                0.0,
                -0.00083218,
                0.0031343,
                0.0018543,
                0.02105921,
                -0.00287067,
                -0.00350567,
                0.00211131,
                -0.01648527,
                0.01162551,
            ],
            [
                0.0,
                0.0,
                -0.048401,
                -0.00038467,
                -0.00066859,
                -0.00465909,
                -0.00230638,
                0.0032778,
                0.0,
                0.0,
                -0.01026985,
                -0.02532982,
                -0.02460769,
                -0.01557121,
                -0.04024391,
                -0.02144066,
                -0.00647471,
                -0.0204918,
                -0.01930845,
                0.0,
                0.0,
                -0.02129827,
                -0.01246468,
                -0.02635055,
                -0.00554038,
                0.0,
                0.0,
                0.01185864,
                -0.00036082,
                -0.00020721,
                -0.00518232,
                -0.00083516,
                -0.01189103,
                -0.01975234,
                -0.013319,
                -0.00083572,
                0.0,
                -0.04191956,
            ],
        ]
    )


# # Use the fixtures in your test functions
# def test_dataframe_with_nan_values(test_df, shap_values):
#     df_with_nans = test_df.copy()
#     df_with_nans.iloc[0, df_with_nans.columns.get_loc("TOTAL_FAILED_HOURS")] = np.nan
#     result_df = select_top_features_for_display(
#         df_with_nans, shap_values
#     )
#     assert not result_df.isnull().values.any()


def test_explanation_output_structure(test_df, predicted_probabilities, shap_values):
    features = test_df.drop(columns="UNIQUE_ID")
    result_df = select_top_features_for_display(
        features, test_df["UNIQUE_ID"], predicted_probabilities, shap_values
    )

    # Basic checks for the result's structure
    assert isinstance(result_df, pd.DataFrame)
    assert "Student ID" in result_df.columns
    assert "Support Score" in result_df.columns
    assert "Top Indicators" in result_df.columns
    assert "SHAP Value" in result_df.columns
    assert "Rank" in result_df.columns

    # Check if the output DataFrame contains the expected number of rows
    expected_rows = len(test_df) * 3
    assert len(result_df) == expected_rows

    sorted_results = result_df.sort_values(["Student ID", "Rank"])
    assert sorted_results[sorted_results["Student ID"] == 1][
        "Top Indicators"
    ].tolist() == [
        "BOTH_FULL_PART_TIME_TERMS",
        "CREDITS_ENROLLED_LAST_3_SEM",
        "PELL_FLAG_COUNT",
    ]
    assert sorted_results[sorted_results["Student ID"] == 2][
        "Top Indicators"
    ].tolist() == [
        "BOTH_FULL_PART_TIME_TERMS",
        "IR_TOTAL_FA_AWD_AMT_SUM",
        "PELL_FLAG_COUNT",
    ]
    assert sorted_results[sorted_results["Student ID"] == 3][
        "Top Indicators"
    ].tolist() == [
        "BOTH_FULL_PART_TIME_TERMS",
        "IS_JUSTICE_ACADEMY",
        "CREDITS_ENROLLED_LAST_3_SEM",
    ]
