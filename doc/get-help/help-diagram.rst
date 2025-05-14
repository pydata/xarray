Getting Help
============

Navigating the wealth of resources available for Xarray can be overwhelming.
We've created this flow chart to help guide you towards the best way to get help, depending on what you're working towards.

Also be sure to check out our "FAQ" and "How do I..." pages in this section for solutions to common questions.

A major strength of Xarray is in the user community. Sometimes you might not have a concrete question by would simply like to connect with other Xarray users. We have a few

We look forward to hearing from you!

.. mermaid::
    :config: {"theme": "default", "fontSize":"16pt"}
    :alt: Flowchart illustrating the different ways to access help using or contributing to Xarray.

    flowchart TD
        intro[Welcome to Xarray! How can we help?]:::quesNodefmt
        usage(["fa:fa-chalkboard-user <a href="https://tutorial.xarray.dev">Xarray Tutorial</a>
            fab:fa-readme <a href="https://docs.xarray.dev">Xarray Docs</a>
            fab:fa-google fab:fa-stack-overflow <a href="https://stackoverflow.com/questions/tagged/python-xarray">Google/Stack Exchange</a>
            fa:fa-robot Ask AI/a Language Learning Model (LLM)"]):::ansNodefmt
        API([fab:fa-readme <a href="https://docs.xarray.dev">Xarray Docs</a>
            fab:fa-readme Extension's docs]):::ansNodefmt
        help([fab:fa-github <a href="https://github.com/pydata/xarray/discussions">Xarray Discussions</a>
            fab:fa-discord <a href="https://discord.com/invite/wEKPCt4PDu">Xarray Discord</a>
            fa:fa-users <a href="https://github.com/pydata/xarray/discussions/categories/office-hours">Xarray Office Hours</a>
            fa:fa-globe <a href="https://discourse.pangeo.io">Pangeo Discourse</a>]):::ansNodefmt
        bug([Report and Propose here:
            fab:fa-github <a href="https://github.com/pydata/xarray/issues">Xarray Issues</a>]):::ansNodefmt
        contrib([fa:fa-book-open <a href="https://docs.xarray.dev/en/latest/contribute">Xarray Contributor's Guide</a>]):::ansNodefmt
        pr(["fab:fa-github <a href="https://github.com/pydata/xarray/pulls">Pull Request (PR)</a>"]):::ansNodefmt
        dev([fab:fa-github Comment on your PR
            fa:fa-users <a href="https://docs.xarray.dev/en/stable/contribute/developers-meeting.html">Developer's Meeting</a> ]):::ansNodefmt
        report[Thanks for letting us know!]:::quesNodefmt
        merged[fa:fa-hands-clapping Your PR was merged.
            Thanks for contributing to Xarray!]:::quesNodefmt


        intro -->|How do I use Xarray?| usage
        usage -->|"with extensions (like Dask)"| API

        usage -->|I'd like some more help| help
        intro -->|I found a bug| bug
        intro -->|I'd like to make a small change| contrib
        subgraph bugcontrib["`**Bugs and Contributions**`"]
            bug
            contrib
            bug -->|I just wanted to tell you| report
            bug<-->|I'd like to fix the bug!| contrib
            pr -->|my PR was approved| merged
        end


        intro -->|I wish Xarray could...| bug


        pr <-->|my PR is quiet| dev
        contrib -->pr

        classDef quesNodefmt font-size:18pt,fill:#9DEEF4,stroke:#206C89

        classDef ansNodefmt font-size:14pt,fill:#FFAA05,stroke:#E37F17

        classDef boxfmt fill:#FFF5ED,stroke:#E37F17
        class bugcontrib boxfmt

        linkStyle default font-size:16pt,color:#206C89

.. toctree::
   :maxdepth: 1
   :hidden:

   faq
   howdoi
   socials
