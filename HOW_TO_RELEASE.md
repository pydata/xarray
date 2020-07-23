# How to issue an xarray release in 17 easy steps

Time required: about an hour.

These instructions assume that `upstream` refers to the main repository:
```
$ git remote -v
{...}
upstream        https://github.com/pydata/xarray (fetch)
upstream        https://github.com/pydata/xarray (push)
```

 1. Ensure your master branch is synced to upstream:
     ```sh
     git pull upstream master
     ```
 2. Get a list of contributors with:
    ```sh
    git log "$(git tag --sort="v:refname" | sed -n 'x;$p').." --format=%aN | sort -u | perl -pe 's/\n/$1, /'
    ```
    or by substituting the _previous_ release in {0.X.Y-1}:
    ```sh
    git log v{0.X.Y-1}.. --format=%aN | sort -u | perl -pe 's/\n/$1, /'
    ```
    Add these into `whats-new.rst` somewhere :)
 2. Write a release summary: ~50 words describing the high level features. This
    will be used in the release emails, tweets, GitHub release notes, etc. 
 3. Look over whats-new.rst and the docs. Make sure "What's New" is complete
    (check the date!) and add the release summary at the top.
    Things to watch out for:
    - Important new features should be highlighted towards the top.
    - Function/method references should include links to the API docs.
    - Sometimes notes get added in the wrong section of whats-new, typically
      due to a bad merge. Check for these before a release by using git diff,
      e.g., `git diff v{0.X.Y-1} whats-new.rst` where {0.X.Y-1} is the previous
      release.
 4. If possible, open a PR with the release summary and whatsnew changes.
 4. After merging, again ensure your master branch is synced to upstream:
     ```sh
     git pull upstream master
     ```
 4. If you have any doubts, run the full test suite one final time!
      ```sh
      pytest
      ```
 5. Check that the ReadTheDocs build is passing.
 6. On the master branch, commit the release in git:
      ```s
      git commit -am 'Release v{0.X.Y}'
      ```
 7. Tag the release:
      ```sh
      git tag -a v{0.X.Y} -m 'v{0.X.Y}'
      ```
 8. Build source and binary wheels for PyPI:
      ```sh
      git clean -xdf  # this deletes all uncommitted changes!
      python setup.py bdist_wheel sdist
      ```
 9. Use twine to check the package build:
      ```sh
      twine check dist/xarray-{0.X.Y}*
      ```
10. Use twine to register and upload the release on PyPI. Be careful, you can't
    take this back!
      ```sh
      twine upload dist/xarray-{0.X.Y}*
      ```
    You will need to be listed as a package owner at
    https://pypi.python.org/pypi/xarray for this to work.
11. Push your changes to master:
      ```sh
      git push upstream master
      git push upstream --tags
      ```
12. Update the stable branch (used by ReadTheDocs) and switch back to master:
     ```sh
      git checkout stable
      git rebase master
      git push --force upstream stable
      git checkout master
     ```
    It's OK to force push to 'stable' if necessary. (We also update the stable 
    branch with `git cherry-pick` for documentation only fixes that apply the 
    current released version.)
13. Add a section for the next release {0.X.Y+1} to doc/whats-new.rst:
     ```
     .. _whats-new.{0.X.Y+1}:

     v{0.X.Y+1} (unreleased)
     ---------------------

     Breaking changes
     ~~~~~~~~~~~~~~~~


     New Features
     ~~~~~~~~~~~~


     Bug fixes
     ~~~~~~~~~


     Documentation
     ~~~~~~~~~~~~~


     Internal Changes
     ~~~~~~~~~~~~~~~~
     ```
14. Commit your changes and push to master again:
      ```sh
      git commit -am 'New whatsnew section'
      git push upstream master
      ```
    You're done pushing to master!
15. Issue the release on GitHub. Click on "Draft a new release" at
    https://github.com/pydata/xarray/releases. Type in the version number
    and paste the release summary in the notes.
16. Update the docs. Login to https://readthedocs.org/projects/xray/versions/
    and switch your new release tag (at the bottom) from "Inactive" to "Active".
    It should now build automatically.
17. Issue the release announcement to mailing lists & Twitter. For bug fix releases, I 
    usually only email xarray@googlegroups.com. For major/feature releases, I will email a broader
    list (no more than once every 3-6 months):
      - pydata@googlegroups.com
      - xarray@googlegroups.com
      - numpy-discussion@scipy.org
      - scipy-user@scipy.org
      - pyaos@lists.johnny-lin.com

    Google search will turn up examples of prior release announcements (look for
    "ANN xarray").

## Note on version numbering

We follow a rough approximation of semantic version. Only major releases (0.X.0)
should include breaking changes. Minor releases (0.X.Y) are for bug fixes and
backwards compatible new features, but if a sufficient number of new features
have arrived we will issue a major release even if there are no compatibility
breaks.

Once the project reaches a sufficient level of maturity for a 1.0.0 release, we
intend to follow semantic versioning more strictly.
