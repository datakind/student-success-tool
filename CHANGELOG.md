# CHANGELOG

## 0.1.1 (2025-02)

- Added "project config" files for consolidating and storing most of the necessary parameters for tailoring general functionality to individual schools' needs (PR #44 #50 #56 #57)
- Extended PDP template notebooks to cover model training and inference (PR #42 #46 #48 #59 #60 #61 #64)
- Improved structure and accessibility of client-facing data output files (PR #43 #53 #58)
- Improved modeling dataset standardization and cleanup with cleaner dtypes, consistent column names, and fewer opportunities for accidental data leakage (PR #38 #45 #49)
- Extended data schemas to cover post-raw transformations of the datasets (PR #36)
- Added new features, including "pre-cohort" and "num courses in study area" (PR #39)
- Added functionality for modeling-adjacent tasks, such as splitting datasets and computing sample weights (PR #41)
- Fixed various bugs and weirdness in PDP synthetic data generation (PR #47 #52)
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
