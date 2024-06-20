Getting help-diagram
====================

Navigating the wealth of resources available for Xarray can be overwhelming.
We've created this flow chart to help guide you towards the best way to get help, depending on what you're working towards.
The links to each resource are provided below the diagram.
Regardless of how you interact with us, we're always thrilled to hear from you!

.. mermaid::
    ---
    title: Navigating Xarray Resources
    ---

    flowchart TD
        intro[Welcome to Xarray! How can we help?]:::quesNodefmt
        usage([fa:fa-chalkboard-user Xarray Tutorials
            fab:fa-readme Xarray Docs
            fab:fa-google Google/fab:fa-stack-overflow Stack Exchange]):::ansNodefmt
        API([fab:fa-readme Xarray Docs
            fab:fa-readme extension's docs]):::ansNodefmt
        help([fab:fa-github Xarray Discussions
            fab:fa-discord Xarray Discord
            fa:fa-users Xarray Office Hours
            fa:fa-globe Pangeo Discourse]):::ansNodefmt
        bug([Report and Propose here:
            fab:fa-github Xarray Issues]):::ansNodefmt
        contrib([fa:fa-book-open Xarray Contributor's Guide]):::ansNodefmt
        pr(["fab:fa-github Pull Request (PR)"]):::ansNodefmt
        dev([fab:fa-github '@' Ping a Maintainer
            fa:fa-users Developer's Meeting]):::ansNodefmt
        report[Thanks for letting us know!]:::quesNodefmt
        merged[Woohoo! Your PR was merged!]:::quesNodefmt


        intro -->|How do I use Xarray?| usage
        usage -->|"with extensions (like Dask)"| API

        usage -->|I'd like some more help| help
        intro -->|I found a bug or typo| bug
        subgraph bugcontrib[Bugs and Contributions]
            bug
            contrib
            bug -->|I just wanted to tell you| report
            bug<-->|I'd like to fix the bug!| contrib
            bug<-->|my plan was approved!| contrib

        end


        intro -->|I wish Xarray could...| bug

        pr -->|my PR was approved| merged

        pr -->|my PR is quiet| dev
        bug -->|my issue is quiet| dev
        contrib -->pr

        classDef quesNodefmt fill:#9DEEF4,stroke:#206C89

        classDef ansNodefmt fill:#FFAA05,stroke:#E37F17

        classDef boxfmt fill:#FFF5ED,stroke:#E37F17
        class bugcontrib boxfmt

        linkStyle default font-size:20pt,color:#206C89


- `Xarray Tutorials <https://tutorial.xarray.dev/>`__
- `Xarray Docs <https://docs.xarray.dev/en/stable/>`__
- `Google/Stack Exchange <https://stackoverflow.com/questions/tagged/python-xarray>`__
- `Xarray Discussions <https://github.com/pydata/xarray/discussions>`__
- `Xarray Discord <https://discord.com/invite/wEKPCt4PDu>`__
- `Xarray Office Hours <https://github.com/pydata/xarray/discussions/categories/office-hours>`__
- `Pangeo Discourse <https://discourse.pangeo.io/>`__
- `Xarray Issues <https://github.com/pydata/xarray/issues>`__
- `Xarray Contributors Guide <https://docs.xarray.dev/en/stable/contributing.html>`__
- `Developer's Meeting <https://docs.xarray.dev/en/stable/developers-meeting.html>`__
