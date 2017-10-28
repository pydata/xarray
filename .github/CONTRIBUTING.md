# Contributing to xarray

## Usage questions

The best places to submit questions about how to use xarray are
[Stack Overflow](https://stackoverflow.com/questions/tagged/python-xarray) and
the [xarray Google group](https://groups.google.com/forum/#!forum/xarray).

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, xarray version and python version. Whenever possible, please
also include a brief, self-contained code example that demonstrates the problem.

## Contributing code

Thanks for your interest in contributing code to xarray!

- If you are new to Git or Github, please take a minute to read through a few tutorials
  on [Git](https://git-scm.com/docs/gittutorial) and [GitHub](https://guides.github.com/).
- The basic workflow for contributing to xarray is:
  1. [Fork](https://help.github.com/articles/fork-a-repo/) the xarray repository
  2. [Clone](https://help.github.com/articles/cloning-a-repository/) the xarray repository to create a local copy on your computer:
    ```
    git clone git@github.com:${user}/xarray.git
    cd xarray
    ```
  3. Create a branch for your changes
    ```
    git checkout -b name-of-your-branch
    ```      
  4. Make change to your local copy of the xarray repository
  5. Commit those changes
    ```
    git add file1 file2 file3
    git commit -m 'a descriptive commit message'
    ```
  6. Push your updated branch to your fork
    ```
    git push origin name-of-your-branch
    ```
  7. [Open a pull request](https://help.github.com/articles/creating-a-pull-request/) to the pydata/xarray repository.
