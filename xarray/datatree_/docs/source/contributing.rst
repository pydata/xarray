========================
Contributing to Datatree
========================

Contributions are highly welcomed and appreciated.  Every little help counts,
so do not hesitate!

.. contents:: Contribution links
   :depth: 2

.. _submitfeedback:

Feature requests and feedback
-----------------------------

Do you like Datatree? Share some love on Twitter or in your blog posts!

We'd also like to hear about your propositions and suggestions.  Feel free to
`submit them as issues <https://github.com/TomNicholas/datatree>`_ and:

* Explain in detail how they should work.
* Keep the scope as narrow as possible.  This will make it easier to implement.

.. _reportbugs:

Report bugs
-----------

Report bugs for Datatree in the `issue tracker <https://github.com/TomNicholas/datatree>`_.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and Datatree
  version.
* Detailed steps to reproduce the bug.

If you can write a demonstration test that currently fails but should pass
(xfail), that is a very useful commit to make as well, even if you cannot
fix the bug itself.

.. _fixbugs:

Fix bugs
--------

Look through the `GitHub issues for bugs <https://github.com/TomNicholas/datatree/labels/type:%20bug>`_.

Talk to developers to find out how you can fix specific bugs.

Write documentation
-------------------

Datatree could always use more documentation. What exactly is needed?

* More complementary documentation.  Have you perhaps found something unclear?
* Docstrings.  There can never be too many of them.
* Blog posts, articles and such -- they're all very appreciated.

You can also edit documentation files directly in the GitHub web interface,
without using a local copy. This can be convenient for small fixes.

To build the documentation locally, you first need to install the following
tools:

- `Sphinx <https://github.com/sphinx-doc/sphinx>`__
- `sphinx_rtd_theme <https://github.com/readthedocs/sphinx_rtd_theme>`__
- `sphinx-autosummary-accessors <https://github.com/xarray-contrib/sphinx-autosummary-accessors>`__

You can then build the documentation with the following commands::

    $ cd docs
    $ make html

The built documentation should be available in the ``docs/_build/`` folder.

.. _`pull requests`:
.. _pull-requests:

Preparing Pull Requests
-----------------------

#. Fork the
   `Datatree GitHub repository <https://github.com/TomNicholas/datatree>`__.  It's
   fine to use ``Datatree`` as your fork repository name because it will live
   under your user.

#. Clone your fork locally using `git <https://git-scm.com/>`_ and create a branch::

    $ git clone git@github.com:{YOUR_GITHUB_USERNAME}/Datatree.git
    $ cd Datatree

    # now, to fix a bug or add feature create your own branch off "master":

    $ git checkout -b your-bugfix-feature-branch-name master

#. Install `pre-commit <https://pre-commit.com>`_ and its hook on the Datatree repo::

     $ pip install --user pre-commit
     $ pre-commit install

   Afterwards ``pre-commit`` will run whenever you commit.

   https://pre-commit.com/ is a framework for managing and maintaining multi-language pre-commit hooks
   to ensure code-style and code formatting is consistent.

#. Install dependencies into a new conda environment::

    $ conda env update -f ci/environment.yml

#. Run all the tests

   Now running tests is as simple as issuing this command::

    $ conda activate datatree-dev
    $ pytest --junitxml=test-reports/junit.xml --cov=./ --verbose

   This command will run tests via the "pytest" tool.

#. You can now edit your local working copy and run the tests again as necessary. Please follow PEP-8 for naming.

   When committing, ``pre-commit`` will re-format the files if necessary.

#. Commit and push once your tests pass and you are happy with your change(s)::

    $ git commit -a -m "<commit message>"
    $ git push -u

#. Finally, submit a pull request through the GitHub website using this data::

    head-fork: YOUR_GITHUB_USERNAME/Datatree
    compare: your-branch-name

    base-fork: TomNicholas/datatree
    base: master
