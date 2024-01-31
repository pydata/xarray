# README - docs

## Build the documentation locally

```bash
cd docs # From project's root
make clean
rm -rf source/generated # remove autodoc artefacts, that are not removed by `make clean`
make html
```

## Access the documentation locally

Open `docs/_build/html/index.html` in a web browser
