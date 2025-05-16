Getting Help
============

Navigating the wealth of resources available for Xarray can be overwhelming.
We've created this flow chart to help guide you towards the best way to get help, depending on what you're working towards.

Also be sure to check out our "FAQ" and "How do I..." pages in this section for solutions to common questions.

A major strength of Xarray is in the user community. Sometimes you might not have a concrete question by would simply like to connect with other Xarray users. We have a few

We look forward to hearing from you!

.. mermaid::
    :config: {"theme":"base","themeVariables":{"fontSize":"20px","darkMode":"true","primaryColor":"#0e4666","primaryTextColor":"#fff","primaryBorderColor":"#59c7d6","lineColor":"#e28126","secondaryColor":"#8e8d99"}}
    :alt: Flowchart illustrating the different ways to access help using or contributing to Xarray.

    flowchart TD
        intro[Welcome to Xarray! How can we help?]:::quesNodefmt
        usage([fa:fa-chalkboard-user <a href="https://tutorial.xarray.dev">Xarray Tutorial</a>
            fab:fa-readme <a href="https://docs.xarray.dev">Xarray Docs</a>
            fab:fa-google fab:fa-stack-overflow <a href="https://stackoverflow.com/questions/tagged/python-xarray">Google/Stack Exchange</a>
            fa:fa-robot Ask AI/a Language Learning Model]):::ansNodefmt
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

        classDef quesNodefmt font-size:18pt,fill:#17afb4,stroke:#0e4666,stroke-width:2
        classDef ansNodefmt stroke-width:2,color:white
        linkStyle default stroke-width:4

.. toctree::
   :maxdepth: 1
   :hidden:

   faq
   howdoi
   socials
