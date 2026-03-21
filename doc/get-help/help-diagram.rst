Getting Help
============

Navigating the wealth of resources available for Xarray can be overwhelming.
We've created this flow chart to help guide you towards the best way to get help, depending on what you're working towards.

Also be sure to check out our :ref:`faq`. and :ref:`howdoi` pages for solutions to common questions.

A major strength of Xarray is in the user community. Sometimes you might not yet have a concrete question but would simply like to connect with other Xarray users. We have a few accounts on different social platforms for that! :ref:`socials`.

We look forward to hearing from you!

Help Flowchart
--------------
..
   _comment: mermaid Flowcharg "link" text gets secondary color background, SVG icon fill gets primary color

.. raw:: html

    <style>
      /* Ensure PST link colors don't override mermaid text colors */
      .mermaid a {
        color: white;
      }
      .mermaid a:hover {
        color: magenta;
        text-decoration-color: magenta;
      }
      .mermaid a:visited {
        color: white;
        text-decoration-color: white;
      }
    </style>

.. mermaid::
    :config: {"theme":"base","themeVariables":{"fontSize":"20px","primaryColor":"#fff","primaryTextColor":"#fff","primaryBorderColor":"#59c7d6","lineColor":"#e28126","secondaryColor":"#767985"}}
    :alt: Flowchart illustrating the different ways to access help using or contributing to Xarray.

    flowchart TD
        intro[Welcome to Xarray! How can we help?]:::quesNodefmt
        usage([fa:fa-chalkboard-user <a href="https://tutorial.xarray.dev">Xarray Tutorial</a>
            fab:fa-readme <a href="https://docs.xarray.dev">Xarray Docs</a>
            fab:fa-stack-overflow <a href="https://stackoverflow.com/questions/tagged/python-xarray">Stack Exchange</a>
            fab:fa-google <a href="https://www.google.com">Ask Google</a>
            fa:fa-robot Ask AI ChatBot]):::ansNodefmt
        extensions([Extension docs:
            fab:fa-readme <a href="https://docs.dask.org">Dask</a>
            fab:fa-readme <a href="https://corteva.github.io/rioxarray">Rioxarray</a>]):::ansNodefmt
        help([fab:fa-github <a href="https://github.com/pydata/xarray/discussions">Xarray Discussions</a>
            fab:fa-discord <a href="https://discord.com/invite/wEKPCt4PDu">Xarray Discord</a>
            fa:fa-globe <a href="https://discourse.pangeo.io">Pangeo Discourse</a>]):::ansNodefmt
        bug([Let us know:
            fab:fa-github <a href="https://github.com/pydata/xarray/issues">Xarray Issues</a>]):::ansNodefmt
        contrib([fa:fa-book-open <a href="https://docs.xarray.dev/en/latest/contribute">Xarray Contributor's Guide</a>]):::ansNodefmt
        pr([fab:fa-github <a href="https://github.com/pydata/xarray/pulls">Pull Request</a>]):::ansNodefmt
        dev([fab:fa-github Add PR Comment
            fa:fa-users <a href="https://docs.xarray.dev/en/stable/contribute/developers-meeting.html">Attend Developer's Meeting</a> ]):::ansNodefmt
        report[Thanks for letting us know!]:::quesNodefmt
        merged[fa:fa-hands-clapping Thanks for contributing to Xarray!]:::quesNodefmt


        intro -->|How do I use Xarray?| usage
        usage -->|"With extensions (like Dask, Rioxarray, etc.)"| extensions

        usage -->|I still have questions or could use some guidance | help
        intro -->|I think I found a bug| bug
        bug
        contrib
        bug -->|I just wanted to tell you| report
        bug<-->|I'd like to fix the bug!| contrib
        pr -->|my PR was approved| merged


        intro -->|I wish Xarray could...| bug


        pr <-->|my PR is quiet| dev
        contrib -->pr

        classDef quesNodefmt font-size:20pt,fill:#0e4666,stroke:#59c7d6,stroke-width:3
        classDef ansNodefmt font-size:18pt,fill:#4a4a4a,stroke:#17afb4,stroke-width:3
        linkStyle default font-size:16pt,stroke-width:4


Flowchart links
---------------
- `Xarray Tutorials <https://tutorial.xarray.dev/>`__
- `Xarray Docs <https://docs.xarray.dev>`__
- `Stack Exchange <https://stackoverflow.com/questions/tagged/python-xarray>`__
- `Xarray Discussions <https://github.com/pydata/xarray/discussions>`__
- `Xarray Discord <https://discord.com/invite/wEKPCt4PDu>`__
- `Xarray Office Hours <https://github.com/pydata/xarray/discussions/categories/office-hours>`__
- `Pangeo Discourse <https://discourse.pangeo.io/>`__
- `Xarray Issues <https://github.com/pydata/xarray/issues>`__
- :ref:`contributing`
- :ref:`developers-meeting`

.. toctree::
   :maxdepth: 1
   :hidden:

   faq
   howdoi
   socials
