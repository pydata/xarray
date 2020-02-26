How to issue an xarray release in 16 easy steps

Time required: about an hour.

 1. Ensure your master branch is synced to upstream:
      ```
      git pull upstream master
      ```
 2. Look over whats-new.rst and the docs. Make sure "What's New" is complete
    (check the date!) and consider adding a brief summary note describing the
    release at the top.
    Things to watch out for:
    - Important new features should be highlighted towards the top.
    - Function/method references should include links to the API docs.
    - Sometimes notes get added in the wrong section of whats-new, typically
      due to a bad merge. Check for these before a release by using git diff,
      e.g., `git diff v0.X.Y whats-new.rst` where 0.X.Y is the previous
      release.
 3. If you have any doubts, run the full test suite one final time!
      ```
      pytest
      ```
 4. Check that the ReadTheDocs build is passing.
 5. On the master branch, commit the release in git:
      ```
      git commit -a -m 'Release v0.X.Y'
      ```
 6. Tag the release:
      ```
      git tag -a v0.X.Y -m 'v0.X.Y'
      ```
 7. Build source and binary wheels for pypi:
      ```
      git clean -xdf  # this deletes all uncommited changes!
      python setup.py bdist_wheel sdist
      ```
 8. Use twine to check the package build:
      ```
      twine check dist/xarray-0.X.Y*
      ```
 9. Use twine to register and upload the release on pypi. Be careful, you can't
    take this back!
      ```
      twine upload dist/xarray-0.X.Y*
      ```
    You will need to be listed as a package owner at
    https://pypi.python.org/pypi/xarray for this to work.
10. Push your changes to master:
      ```
      git push upstream master
      git push upstream --tags
      ```
11. Update the stable branch (used by ReadTheDocs) and switch back to master:
     ```
      git checkout stable
      git rebase master
      git push upstream stable
      git checkout master
     ```
    It's OK to force push to 'stable' if necessary. (We also update the stable 
    branch with `git cherrypick` for documentation only fixes that apply the 
    current released version.)
12. Add a section for the next release (v.X.(Y+1)) to doc/whats-new.rst.
13. Commit your changes and push to master again:
      ```
      git commit -a -m 'New whatsnew section'
      git push upstream master
      ```
    You're done pushing to master!
14. Issue the release on GitHub. Click on "Draft a new release" at
    https://github.com/pydata/xarray/releases. Type in the version number, but
    don't bother to describe it -- we maintain that on the docs instead.
15. Update the docs. Login to https://readthedocs.org/projects/xray/versions/
    and switch your new release tag (at the bottom) from "Inactive" to "Active".
    It should now build automatically.
16. Issue the release announcement! For bug fix releases, I usually only email
    xarray@googlegroups.com. For major/feature releases, I will email a broader
    list (no more than once every 3-6 months):
      - pydata@googlegroups.com
      - xarray@googlegroups.com
      - numpy-discussion@scipy.org
      - scipy-user@scipy.org
      - pyaos@lists.johnny-lin.com

    Google search will turn up examples of prior release announcements (look for
    "ANN xarray").
    You can get a list of contributors with:
    ```
    git log "$(git tag --sort="v:refname" | sed -n 'x;$p').." --format="%aN" | sort -u
    ```
    or by replacing `v0.X.Y` with the _previous_ release in:
    ```
    git log v0.X.Y.. --format="%aN" | sort -u
    ```

Note on version numbering:

We follow a rough approximation of semantic version. Only major releases (0.X.0)
show include breaking changes. Minor releases (0.X.Y) are for bug fixes and
backwards compatible new features, but if a sufficient number of new features
have arrived we will issue a major release even if there are no compatibility
breaks.

Once the project reaches a sufficient level of maturity for a 1.0.0 release, we
intend to follow semantic versioning more strictly.
