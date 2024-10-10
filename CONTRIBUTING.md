# Contributing to SST

Hi! Thanks for your interest in contributing to DataKind's Student Success Tool, we're really excited to see you! In this document we'll try to summarize everything that you need to know to do a good job.

## New contributor guide

To get an overview of the project, please read the [README](README.md) and our [Code of Conduct](./CODE_OF_CONDUCT.md) to keep our community approachable and respectable.


## Getting started
### Creating Issues

If you spot a problem, [search if an issue already exists](https://github.com/datakind/Student-Success-Tool/issues). If a related issue doesn't exist,
you can open a new issue using a relevant [issue form](https://github.com/datakind/Student-Success-Tool/issues/new).

As a general rule, we don’t assign issues to anyone. If you find an issue to work on, you are welcome to open a PR with a fix.

## Making Code changes

### Environment
We use [Poetry](https://github.com/python-poetry/poetry/tree/master) for package management. To get up and running quickly, install the environment with:
```
poetry install --no-interaction
```

### GitHub Workflow

As many other open source projects, we use the famous [gitflow](https://nvie.com/posts/a-successful-git-branching-model/) to manage our branches.

Summary of our git branching model:
- Get all the latest work from the upstream `datakind/Student-Success-Tool` repository
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

Some branch definitions:
- `main`: always stable and release ready branch
- `develop`: default branch, contains latest features and fixes, on which developers should orient
- `feature-*`: branches for feature development and dependency updates
- `release-*`: branches for release versions
- `hotfix-*`: branches for release or production fixes
- `refactor-*`: branches for semantic changes
- `institution_id-*`: For partner-specific analyses and programming, use the institution ID as the branch prefix.

Pull Request guidelines:
- For each pull request, there should be an associated Asana ticket linked.
- Except for quick fixes, the develop branch should be started from develop and merged back.
- Hotfix branches should be started from main and must be merged back to main and develop. It is also possible to start hotfix branches from a release branch and merged back to main, develop, and the release branch.
- Any release branch should start from the develop branch. Starting a release branch unblocks new feature development. Merging a release branch to main indicates a new version in production.


### Tips

- Write [helpful commit messages](https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message)
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


## Versioning

Each release should be documented in the HISTORY.md file.

Releases to `main` should be tagged with a [semantic version number](https://semver.org/). If you use git flow hooks, this is as simple as `git flow release start 1.X.Y`.  When you finish the release, the hook will tag your production release appropriately.  If you don't use that, you need to checkout `main`, then `git tag 1.X.Y -m "some message about this release"`.

Semver format:
MAJOR.MINOR.BUGFIX

* `MAJOR`: Means breaking changes happen from one version to the next.
* `MINOR`: Extends the existing code, but should be backwards compatible with old code.
* `BUGFIX`: Doesn't add much new, but fixes issues along the way.
