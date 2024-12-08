# CHANGELOG

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
