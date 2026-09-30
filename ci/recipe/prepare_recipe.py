#!/usr/bin/env python

import pathlib
import textwrap

import vcs_versioning


def find_pyproject(dir):
    while dir != dir.root:
        path = dir / "pyproject.toml"
        if path.is_file():
            return path

        dir = dir.parent

    raise RuntimeError("pyproject.toml can't be found")


def main():
    cwd = pathlib.Path(__file__).parent
    pyproject_path = find_pyproject(cwd)
    root = pyproject_path.parent

    pyproject = vcs_versioning.PyProjectData.from_file(pyproject_path)
    version = vcs_versioning.infer_version_string("xarray", pyproject)

    recipe_root = root / "ci/recipe"

    template_path = recipe_root / "recipe_template.yaml"
    template = template_path.read_text()

    context = textwrap.dedent(
        f"""\
        context:
          name: xarray
          version: {version}
        """.rstrip()
    )
    recipe = "\n".join([context, "", template])  # noqa: FLY002
    recipe_path = recipe_root / "recipe.yaml"
    recipe_path.write_text(recipe)


if __name__ == "__main__":
    main()
