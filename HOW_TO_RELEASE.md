# How to issue an xarray release in 16 easy steps

Time required: about an hour.

These instructions assume that `upstream` refers to the main repository:

```sh
$ git remote -v
{...}
upstream        https://github.com/pydata/xarray (fetch)
upstream        https://github.com/pydata/xarray (push)
```

<!-- markdownlint-disable MD031 -->

1.  Ensure your main branch is synced to upstream:
    ```sh
    git switch main
    git pull upstream main
    ```
2.  Add a list of contributors.
    First fetch all previous release tags so we can see the version number of the last release was:

    ```sh
    git fetch upstream --tags
    ```

    Then run

    ```sh
    python ci/release_contributors.py
    ```

    (needs `gitpython` and `toolz` / `cytoolz`)

    and copy the output.

3.  Write a release summary: ~50 words describing the high level features. This
    will be used in the release emails, tweets, GitHub release notes, etc.
4.  Look over whats-new.rst and the docs. Make sure "What's New" is complete
    (check the date!) and add the release summary at the top.
    Things to watch out for:
    - Important new features should be highlighted towards the top.
    - Function/method references should include links to the API docs.
    - Sometimes notes get added in the wrong section of whats-new, typically
      due to a bad merge. Check for these before a release by using git diff,
      e.g., `git diff v{YYYY.MM.X-1} whats-new.rst` where {YYYY.MM.X-1} is the previous
      release.
5.  Open a PR with the release summary and whatsnew changes; in particular the
    release headline should get feedback from the team on what's important to include.
6.  After merging, again ensure your main branch is synced to upstream:
    ```sh
    git pull upstream main
    ```
7.  If you have any doubts, run the full test suite one final time!
    ```sh
    pytest
    ```
8.  Check that the [ReadTheDocs build](https://readthedocs.org/projects/xray/) is passing on the `latest` build version (which is built from the `main` branch).
9.  Issue the release on GitHub. Click on "Draft a new release" at
    <https://github.com/pydata/xarray/releases>. Type in the version number (with a "v")
    and paste the release summary in the notes.
10. This should automatically trigger an upload of the new build to PyPI via GitHub Actions.
    Check this has run [here](https://github.com/pydata/xarray/actions/workflows/pypi-release.yaml),
    and that the version number you expect is displayed [on PyPI](https://pypi.org/project/xarray/)
11. Add a section for the next release {YYYY.MM.X+1} to doc/whats-new.rst (we avoid doing this earlier so that it doesn't show up in the RTD build):

    ```rst
    .. _whats-new.YYYY.MM.X+1:

    vYYYY.MM.X+1 (unreleased)
    -----------------------

    New Features
    ~~~~~~~~~~~~


    Breaking changes
    ~~~~~~~~~~~~~~~~


    Deprecations
    ~~~~~~~~~~~~


    Bug fixes
    ~~~~~~~~~


    Documentation
    ~~~~~~~~~~~~~


    Internal Changes
    ~~~~~~~~~~~~~~~~

    ```

12. Commit your changes and push to main again:

    ```sh
    git commit -am 'New whatsnew section'
    git push upstream main
    ```

    You're done pushing to main!

13. Update the version available on pyodide:

    - Open the PyPI page for [Xarray downloads](https://pypi.org/project/xarray/#files)
    - Edit [`pyodide/packages/xarray/meta.yaml`](https://github.com/pyodide/pyodide/blob/main/packages/xarray/meta.yaml) to update the
      - version number
      - link to the wheel (under "Built Distribution" on the PyPI page)
      - SHA256 hash (Click "Show Hashes" next to the link to the wheel)
    - Open a pull request to pyodide

14. Issue the release announcement to mailing lists & Twitter. For bug fix releases, I
    usually only email xarray@googlegroups.com. For major/feature releases, I will email a broader
    list (no more than once every 3-6 months):

    - pydata@googlegroups.com
    - xarray@googlegroups.com
    - numpy-discussion@scipy.org
    - scipy-user@scipy.org
    - pyaos@lists.johnny-lin.com

    Google search will turn up examples of prior release announcements (look for
    "ANN xarray").
    Some of these groups require you to be subscribed in order to email them.

<!-- markdownlint-enable MD013 -->

## Note on version numbering

As of 2022.03.0, we utilize the [CALVER](https://calver.org/) version system.
Specifically, we have adopted the pattern `YYYY.MM.X`, where `YYYY` is a 4-digit
year (e.g. `2022`), `0M` is a 2-digit zero-padded month (e.g. `01` for January), and `X` is the release number (starting at zero at the start of each month and incremented once for each additional release).
