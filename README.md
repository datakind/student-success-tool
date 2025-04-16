# DataKind's Student Success Tool (SST)

Customized and easily actionable insights for data-assisted advising, at no cost

Data-assisted advising helps advisors use their limited time to more efficiently identify and reach out to those most in need of help.
Using the Student Success Tool to implement data-assisted advising, John Jay College has reported a 32% increase in senior graduation rates in two years via their CUSP program.
Based on the success of this implementation, DataKind is supported by Google.org to develop this solution with additional postsecondary institutions, at no institutional cost.
This repo is where the google.org fellows team will collaborate with DataKind to develop and ultimately share the open source components of the tool.

### DataKind's Product Principles

- Transparent: Our features and models will be openly shared with the institution, so you can know exactly what variables are leading to identifying those student most at risk of non graduation. Our end-to-end tool code will be openly shared in this github repo.
- Dedicated to bias reduction: We use bias-reducing techniques and regularly review our implementations for fairness and equity.
- Humans in the loop by design: Our interventions are designed to be additive to the student experience, and all algorithms are implemented through human actors (advisors).

## Model Training and Prediction Workflow

![Student Success Tool (SST) model training and implementation workflow (4)](https://github.com/user-attachments/assets/1a3816bc-acd5-4b53-ad92-929a66bebbac)

## What's in this repo?

Current PDP pipeline code: to be built into an actual installable python package
- Base schema: defines the standard data schema for PDP schools, with no customization
- Constants: defined for all schools
- Dataio: ingests the PDP data and restructures it for our workflow
- Features: subpackage for each grouping of features with a function that takes school customization arguments and adds the features to the data you give it as new columns.
- EDA: produces exploratory visualizations, summary statistics, and coorelation analysis for features
- Targets: defines and filters the data based on the student population, modeling checkpoint, and outcome variable
- Dataops: other functions frequently used across the process
- Modeling: AutoML.py is the main code that can be used for running and evaluating models, configured with parameters accepted from the config.yaml
- Tests: unit tests, to be built out into full unit testing suite (possibly fellows can help with this to get us set up for open source)
- Synthetic_data: Code for creating fake data for testing purposes

## Contributing

Please read the [CONTRIBUTING](CONTRIBUTING.md) to learn how to contribute to the tool development.

## Setup

### local machine

1. Install `uv` (instructions [here](https://docs.astral.sh/uv/getting-started/installation)).
1. Install Python (instructions [here](https://docs.astral.sh/uv/guides/install-python)). When running on Databricks, we're constrained to Python 3.11-3.12: `uv python install 3.11`
1. Install this package: `uv pip install -e .`

### databricks notebook

1. Connect notebook to a cluster running Databricks Runtime [15.4 LTS](https://docs.databricks.com/en/release-notes/runtime/15.4lts.html) or [16.x](https://docs.databricks.com/aws/en/release-notes/runtime/16.2).
1. Run the `%pip` magic command, pointing it at one of three places:
    - a local workspace directory: `%pip install ../../../student-success-tool/`
    - a GitHub repo (for a specific branch): `%pip install git+https://github.com/datakind/student-success-tool.git@develop`
    - public PyPI: `%pip install student-success-tool == x.y.z`
1. Restart Python: `dbutils.library.restartPython()` or `%restart_python`

## Package Management

### releases

1. Ensure that all changes (features, bug fixes, etc.) to be included in the release have been merged into the `develop` branch.
2. Create a new feature branch based off `develop` that includes two release-specific changes:
    - bump the `project.version` attribute in the package's `pyproject.toml` file to the desired version; follow [SemVer conventions](https://semver.org)
    - add an entry in `CHANGELOG.md` for the specified version, with a manually-curated summary of the changes included in the release, optionally including call-outs to specific PRs for reference
3. Merge the above PR into `develop`, then open a new PR to merge all changes in `develop` into the `main` branch; merge it
4. Go to the GitHub repo's [Releases](https://github.com/datakind/student-success-tool/releases) page, then click the "draft a new release" button
    - choose a tag; it should be formatted as "v[VERSION]", for example "v0.2.0"
    - choose `main` as the target branch
    - enter a release title; it could be as simple as "v[VERSION]"
    - copy-paste the changelog entry for this version into the "describe this release" text input
    - click the "publish release" button
5. Check the repo's [GitHub actions](https://github.com/datakind/student-success-tool/actions) to ensure that the `publish` workflow runs, and once it completes, check the package's [PyPI page](https://pypi.org/project/student-success-tool) to ensure that the new version is live

Et voilà, a new version has been released! :tada:
