# CHANGELOG

## 0.2.0 (2025-04)

- Extended and modularized functionality into several new subpackages, with `pdp` as a secondary level, to make imports more intuitive and allow for extending code to support other 3rd-party data formats
    - Consolidated functionality for reading/writing data in `dataio` (PR #71)
    - Added new `targets` functions -- corresponding to and alongside existing functions -- with a more consistent, configurable, and modularized API (PR #76)
    - Added `selection` and `checkpoints` subpackages for selecting student populations for modeling and identifying "checkpoint" student-terms for prediction, respectively, separate from target calculation (PR #77)
    - Added general-purpose `converters` functions for transforming raw data such that it's parseable by the corresponding data schemas (PR #122, #134)
    - Consolidated grab-bag "utility" functionality into a `utils` subpackage, and added more helpful functions to it (PR #86, #99, #100, #102)
- Improved handling for raw PDP data with different student id columns, for columns added to raw PDP data as of 2025-01, and with more forgiving data validation to reduce the need for school-specific overrides (PR #84, #88, #90, #95)
- Improved feature engineering with new configuration for "core terms" (and better associated features), simpler / more interpretable feature formulations, and a handful of new features (PR #101, #111, #120)
- Added new methodology ("false negative parity rate") for model bias evaluation and incorporated it into model training process (PR #118, #124, #128, #129)
- Fixed various rough edges and filled various gaps for better devx, including better type annotations, utility functions for working in Databricks, model naming standards, consistent parameter and function names (PR #75, #83, #115, #127)
- Refactored template notebooks to better leverage project configs, more closely resemble actual school-specific "pipelines", more thoroughly assess data (PR #73, #108, #112, #123)
- Improved calculation and visualization of SHAP model explanations (PR #92, #94, #96, #98)
- Added matching logic to support mapping all* features to human-friendly feature names (PR #104)
- Included formatting in "style" CI workflow to ensure consistency of code, and made GitHub actions safer (PR #72, #106, #107, #109)
- Update setup instructions and add release instructions in readme (PR #117)
- Added proof-of-concept standardized model inference pipeline that runs in Databricks (PR #78, #82, #105, #113, #114, #119)

## 0.1.1 (2025-02)

- Added "project config" files for consolidating and storing most of the necessary parameters for tailoring general functionality to individual schools' needs (PR #44 #50 #56 #57)
- Extended PDP template notebooks to cover model training and inference (PR #42 #46 #48 #59 #60 #61 #64)
- Improved structure and accessibility of client-facing data output files (PR #43 #53 #58)
- Improved modeling dataset standardization and cleanup with cleaner dtypes, consistent column names, and fewer opportunities for accidental data leakage (PR #38 #45 #49)
- Extended data schemas to cover post-raw transformations of the datasets (PR #36)
- Added new features, including "pre-cohort" and "num courses in study area" (PR #39)
- Added functionality for modeling-adjacent tasks, such as splitting datasets and computing sample weights (PR #41)
- Fixed various bugs and weirdness in PDP synthetic data generation (PR #47 #52 #68)
- Updated key dependencies and added support for PY3.12 (PR #40)

## 0.1.0 (2024-11)

- Ported school-agnostic code from private repo, with some refactoring of structure and modest code quality improvements (PR #1 #2 #3 #6 #10)
- Set up Python packaging with `uv` and updated CI workflows (PR #5 #8 #13 #17 #29 #30 #32)
- Extended and improved featurization functionality, including better course grade handling, term- and year-level features, "term diff" features over time (PR: #4 #7 #11 #12 #15 #20 #21 #22 #23)
- Extended and improved target variable functionality, including a new "failure to retain" target and higher-level `make_labeled_dataset()` entry points for each target for developer convenience (PR #24 #33)
- Refactored and better generalized PDP raw data schemas (PR #19 #28)
- Added functionality for generating synthetic PDP and "sample platform" data (PR #9)
- Added generalized "pairwise association" function for comparing variables of various data types (PR #31)
- Added template notebooks for the data assessment/EDA and modeling dataset prep steps of the SST process (PR #26)
- Various minor bugfixes
