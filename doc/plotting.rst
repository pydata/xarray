Plotting
--------

Examples
~~~~~~~~

Here are some quick examples of plotting in xray.

To begin, import numpy, pandas and xray:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xray
    import matplotlib.pyplot as plt

    @savefig plotting_example1.png
    plt.plot((0, 1), (0, 1))

Rules
~~~~~

xray tries to create reasonable plots based on metadata and the array
dimensions.

'In the face of ambiguity, refuse the temptation to guess.'
