# Iris custom benchmarks

To be recognised by ASV, these benchmarks must be packaged and installed in 
line with the
[ASV guidelines](https://asv.readthedocs.io/projects/asv-runner/en/latest/development/benchmark_plugins.html).
This is achieved using the custom build in [install.py](./install.py).

Installation is into the environment where the benchmarks are run (i.e. not
the environment containing ASV + Nox, but the one built to the same
specifications as the Tests environment). This is done via `build_command`
in [asv.conf.json](../asv.conf.json).
