#!/usr/bin/env python
import argparse
import itertools
import pathlib

import yaml
from packaging import version


def extract_versions(config):
    repos = config.get("repos")
    if repos is None:
        raise ValueError("invalid pre-commit configuration")

    extracted_versions = (
        ((hook["id"], version.parse(repo["rev"])) for hook in repo["hooks"])
        for repo in repos
    )
    return dict(itertools.chain.from_iterable(extracted_versions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    parser.add_argument(
        metavar="pre-commit-config", dest="pre_commit_config", type=pathlib.Path
    )
    parser.add_argument("requirements", type=pathlib.Path)
    args = parser.parse_args()

    with args.pre_commit_config.open() as f:
        config = yaml.safe_load(f)

    versions = extract_versions(config)
    mypy_version = versions["mypy"]

    requirements = args.requirements.read_text()
    new_requirements = "\n".join(
        [
            line if not line.startswith("mypy=") else f"mypy={mypy_version}"
            for line in requirements.split("\n")
        ]
    )

    if args.dry:
        separator = "\n" + "—" * 80 + "\n"
        print(
            "contents of the new requirements file:",
            new_requirements,
            sep=separator,
            end=separator,
        )
    else:
        args.requirements.write_text(new_requirements)
