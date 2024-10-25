# PDP Data Analysis

## Contents

- **DataIO:** Functions to read raw PDP course/cohort datasets from disk and optionally validate the data, as well as generic functions for reading/writing data from/to Databricks Delta Tables.
- **Schemas:** Classes used to validate PDP data at notable steps in the pipeline, in PDP-standard "base" forms _only_. School-specific sub-classes that allow for some variation within the official spec are defined in those schools' workflows.
- **DataOps:** Bit of a grab-bag, at the moment! Its main use case is a function to transform raw course and cohort datasets into featurized data aggregated at the student-term level.
- **Features:** Sub-package with modules that compute a wide variety of features at varying units of analysis (student, course, term, section, etc.) based on the raw PDP datasets.
- **Targets:** Sub-package with modules that select eligible students and compute target variables according to a couple of different formulations.
- **Miscellany:** Generic utils functions (`utils.py`), several functions useful in exploratory data analysis (`eda.py`), constant values used across modules (`constants.py`).

## Usage

1. Read raw PDP datasets from CSV files; validate them using schemas.

    ```python
    fpath_cohort = os.path.join(path_volume, "XYZ_STUDENT_SEMESTER_AR_DEIDENTIFIED_123.csv")
    # check if raw data conforms to base schema
    pdp.dataio.read_raw_pdp_cohort_data_from_file(
        fpath_cohort, schema=pdp.schemas.base.RawPDPCohortDataSchema
    )
    # in case of validation errors here, subclass base w/ school-specific schema
    # that correctly handles minor variations, until raw data properly validates
    df_cohort = pdp.dataio.read_raw_pdp_cohort_data_from_file(
        fpath_cohort, schema=pdp.schemas.xyz.RawPDPCohortDataSchema
    )
    ```

    - In case of major raw data incompatibility with the PDP spec, develop a "preprocessor" function that transforms the raw data into a compatible form.

        ```python
        pdp.dataio.read_raw_pdp_cohort_data_from_file(
            fpath_cohort, schema=None, preprocessor_func=fix_my_weird_raw_dataset
        )
        ```

1. Explore the raw data with help from EDA functions. (TODO)

1. Transform and featurize the raw data using DataOps funcs, which call feature-funcs extensively under the hood.

    ```python
    df_student_terms = pdp.dataops.make_student_term_dataset(
        df_cohort,
        df_course,
        min_passing_grade=1.0,
        key_course_subject_areas=["24", "51"],
    )
    ```

1. Select eligible students, and compute target variables based on the featurized data.

    ```python
    df_eligible_students = pdp.targets_v2.failure_to_earn_enough_credits_in_time_from_enrollment.select_eligible_students(
        df_student_terms,
        student_criteria={
            "credential_type_sought_year_1": "Associate's Degree",
            "enrollment_type": "FIRST-TIME",
            "enrollment_intensity_first_term": ["FULL-TIME", "PART-TIME"],
        },
        intensity_time_limits = [
            ("FULL-TIME", 3.0, "year"),
            ("PART-TIME", 6.0, "year"),
        ],
        min_num_credits_checkin=30.0,
    )
    df_eligible_student_terms = pd.merge(
        df_student_terms, df_eligible_students, on="student_guid", how="inner"
    )
    student_targets = pdp.targets_v2.failure_to_earn_enough_credits_in_time_from_enrollment.compute_target_variable(
        df_eligible_student_terms,
        intensity_time_limits = [
            ("FULL-TIME", 3.0, "year"),
            ("PART-TIME", 6.0, "year"),
        ],
        min_num_credits_target=60.0,
    )
    ```

1. Prepare data for modeling with feature selection, subsetting, etc. (TODO)

1. Write modeling dataset to table(s) in Unity Catalog.

    ```python
    write_table_path = f"{catalog}.{write_schema}.labeled_selected_data"
    pdp.dataio.write_data_to_delta_table(df_labeled_selected, write_table_path, spark)
    ```
