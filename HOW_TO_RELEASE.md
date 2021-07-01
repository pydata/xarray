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

 1. Ensure your main branch is synced to upstream:
     ```sh
     git switch main
     git pull upstream main
     ```
 2. Confirm there are no commits on stable that are not yet merged
    ([ref](https://github.com/pydata/xarray/pull/4440)):
     ```sh
     git merge upstream/stable
     ```
 3. Add a list of contributors with:
    ```sh
    git log "$(git tag --sort="v:refname" | tail -1).." --format=%aN | sort -u | perl -pe 's/\n/$1, /'
    ```
    This will return the number of contributors:
    ```sh
    git log $(git tag --sort="v:refname" | tail -1).. --format=%aN | sort -u | wc -l
    ```
 4. Write a release summary: ~50 words describing the high level features. This
    will be used in the release emails, tweets, GitHub release notes, etc.
 5. Look over whats-new.rst and the docs. Make sure "What's New" is complete
    (check the date!) and add the release summary at the top.
    Things to watch out for:
    - Important new features should be highlighted towards the top.
    - Function/method references should include links to the API docs.
    - Sometimes notes get added in the wrong section of whats-new, typically
      due to a bad merge. Check for these before a release by using git diff,
      e.g., `git diff v{0.X.Y-1} whats-new.rst` where {0.X.Y-1} is the previous
      release.
 6. Open a PR with the release summary and whatsnew changes; in particular the
    release headline should get feedback from the team on what's important to include.
 7. After merging, again ensure your main branch is synced to upstream:
     ```sh
     git pull upstream main
     ```
 8. If you have any doubts, run the full test suite one final time!
      ```sh
      pytest
      ```
 9. Check that the ReadTheDocs build is passing.
10. Issue the release on GitHub. Click on "Draft a new release" at
    <https://github.com/pydata/xarray/releases>. Type in the version number (with a "v")
    and paste the release summary in the notes.
11. This should automatically trigger an upload of the new build to PyPI via GitHub Actions.
    Check this has run [here](https://github.com/pydata/xarray/actions/workflows/pypi-release.yaml),
    and that the version number you expect is displayed [on PyPI](https://pypi.org/project/xarray/)
12. Update the stable branch (used by ReadTheDocs) and switch back to main:
     ```sh
      git switch stable
      git rebase main
      git push --force upstream stable
      git switch main
     ```
    You may need to first fetch it with `git fetch upstream`,
    and check out a local version with `git checkout -b stable upstream/stable`.

    It's OK to force push to `stable` if necessary. (We also update the stable
    branch with `git cherry-pick` for documentation only fixes that apply the
    current released version.)
13. Add a section for the next release {0.X.Y+1} to doc/whats-new.rst:
     ```rst
     .. _whats-new.0.X.Y+1:

     v0.X.Y+1 (unreleased)
     ---------------------

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
14. Commit your changes and push to main again:
      ```sh
      git commit -am 'New whatsnew section'
      git push upstream main
      ```
    You're done pushing to main!

15. Update the docs. Login to <https://readthedocs.org/projects/xray/versions/>
    and switch your new release tag (at the bottom) from "Inactive" to "Active".
    It should now build automatically.
16. Issue the release announcement to mailing lists & Twitter. For bug fix releases, I
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

We follow a rough approximation of semantic version. Only major releases (0.X.0)
should include breaking changes. Minor releases (0.X.Y) are for bug fixes and
backwards compatible new features, but if a sufficient number of new features
have arrived we will issue a major release even if there are no compatibility
breaks.

Once the project reaches a sufficient level of maturity for a 1.0.0 release, we
intend to follow semantic versioning more strictly.
