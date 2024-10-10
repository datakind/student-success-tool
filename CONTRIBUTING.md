# Contributing to DOT

Hi! Thanks for your interest in contributing to this project, we're really excited to see you! In this document we'll try to
summarize everything that you need to know to do a good job.

## New contributor guide

To get an overview of the project, please read the [README](README.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.


## Getting started
### Creating Issues

If you spot a problem, [search if an issue already exists](https://github.com/datakind/Data-Observation-Toolkit/issues). If a related issue doesn't exist,
you can open a new issue using a relevant [issue form](https://github.com/datakind/Data-Observation-Toolkit/issues/new).

As a general rule, we don’t assign issues to anyone. If you find an issue to work on, you are welcome to open a PR with a fix.

## Making Code changes

## Development setup

1. Create and activate conda environment
2. Install pre-commit hooks
3. Run unit tests

### Create and activate conda environment

For the following you will need [miniconda](https://docs.conda.io/en/main/miniconda.html) installed.

```bash
# From the repo root directory
conda env create
conda activate appenv
```

If successful, you should see:
```bash
# To activate this environment, use
#
#     $ conda activate appenv
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### Install pre-commit hooks

```bash
# From the application/backend directory
pre-commit install
```

If successful, you should see:
```
pre-commit installed at .git/hooks/pre-commit
```

Now whenever you commit any file, all the hooks listed in `.pre-commit-config.yaml` will run against the commited file(s). For example, upon adding sample files to the `data` folder, you would see in your terminal:
```
(bwdc) jtan@Jonathans-MacBook-Pro-2 backend % git add data
(bwdc) jtan@Jonathans-MacBook-Pro-2 backend % git commit -m "Add sample data used in template config"
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check Yaml...........................................(no files to check)Skipped
Check for merge conflicts................................................Passed
Check Toml...........................................(no files to check)Skipped
Don't commit to branch...................................................Passed
Detect Private Key.......................................................Passed
absolufy-imports.....................................(no files to check)Skipped
Lint Dockerfiles.....................................(no files to check)Skipped
black................................................(no files to check)Skipped
isort................................................(no files to check)Skipped
flake8...............................................(no files to check)Skipped
[JT-migrate-existing-spatial-targeting-code aabfc50] Add sample data used in template config
 3 files changed, 224 insertions(+)
 create mode 100644 application/backend/data/sample_census_tracts.geojson
 create mode 100644 application/backend/data/sample_event_sites.geojson
 create mode 100644 application/backend/data/sample_housing_loss_indicator_data.csv
```

### Run unit tests

To run all tests, from the root directory, run:
```bash
python -m pytest
```

### GitHub Workflow

As many other open source projects, we use the famous
[gitflow](https://nvie.com/posts/a-successful-git-branching-model/) to manage our
branches.

Summary of our git branching model:
- Get all the latest work from the upstream `datakind/Data-Observation-Toolkit` repository
  (`git checkout main`)
- Create a new branch off with a descriptive name (for example:
  `feature/new-test-macro`, `bugfix/bug-when-uploading-results`). You can
  do it with (`git checkout -b <branch name>`)
- Make your changes and commit them locally  (`git add <changed files>>`,
  `git commit -m "Add some change" <changed files>`). Whenever you commit, the self-tests
  and code quality will kick in; fix anything that gets broken
- Push to your branch on GitHub (with the name as your local branch:
  `git push origin <branch name>`). This will output a URL for creating a Pull Request (PR)
- Create a pull request by opening the URL a browser. You can also create PRs in the GitHub
  interface, choosing your branch to merge into main
- Wait for comments and respond as-needed
- Once PR review is complete, your code will be merged. Thanks!!


### Tips

- Write [helpful commit
  messages](https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message)
- Anything in your branch must have no failing tests. You can check by looking at your PR
  online in GitHub
- Never use `git add .`: it can add unwanted files;
- Avoid using `git commit -a` unless you know what you're doing;
- Check every change with `git diff` before adding them to the index (stage
  area) and with `git diff --cached` before committing;
- If you have push access to the main repository, please do not commit directly
  to `dev`: your access should be used only to accept pull requests; if you
  want to make a new feature, you should use the same process as other
  developers so your code will be reviewed.


## Code Guidelines

- Use [PEP8](https://www.python.org/dev/peps/pep-0008/);
- Write tests for your new features (please see "Tests" topic below);
- Always remember that [commented code is dead
  code](https://www.codinghorror.com/blog/2008/07/coding-without-comments.html);
- Name identifiers (variables, classes, functions, module names) with readable
  names (`x` is always wrong);
- When manipulating strings, we prefer either [f-string
  formatting](https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals)
  (f`'{a} = {b}'`) or [new-style
  formatting](https://docs.python.org/library/string.html#format-string-syntax)
  (`'{} = {}'.format(a, b)`), instead of the old-style formatting (`'%s = %s' % (a, b)`);
- You will know if any test breaks when you commit, and the tests will be run
  again in the continuous integration pipeline (see below);

## Tests

You should write tests for every feature you add or bug you solve in the code.
Having automated tests for every line of our code lets us make big changes
without worries: there will always be tests to verify if the changes introduced
bugs or lack of features. If we don't have tests we will be blind and every
change will come with some fear of possibly breaking something.

For a better design of your code, we recommend using a technique called
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development),
where you write your tests **before** writing the actual code that implements
the desired feature.

You can type `pytest` to run your tests, no matter which type of test it is.


## Continuous Integration

We use [GitHub Actions](https://github.com/datakind/Data-Observation-Toolkit/actions)
for continuous integration.
See [here](https://docs.github.com/en/actions) for GitHub's documentation.

The [`.github/workflows/lint.yml`](.github/workflows/ci.yml) file configures the CI.
