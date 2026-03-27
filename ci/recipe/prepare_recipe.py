#!/usr/bin/env python

import pathlib
import textwrap

import git
import packaging.version
from tlz.itertoolz import last


def dev_version(most_recent_release):
    v = packaging.version.parse(str(most_recent_release))

    next_version = (v.major, v.minor, v.micro + 1)
    return str(v.__replace__(release=next_version, dev=0))


def main():
    repo = git.Repo(".")
    root = pathlib.Path(repo.working_dir)

    most_recent_release = last(list(repo.tags))
    version = dev_version(most_recent_release)
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
