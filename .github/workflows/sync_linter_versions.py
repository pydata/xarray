#!/usr/bin/env python
import argparse
import itertools
import pathlib
import re

import yaml
from packaging import version
from packaging.requirements import Requirement

operator_re = re.compile("=+")


def extract_versions(config):
    repos = config.get("repos")
    if repos is None:
        raise ValueError("invalid pre-commit configuration")

    extracted_versions = (
        ((hook["id"], version.parse(repo["rev"])) for hook in repo["hooks"])
        for repo in repos
    )
    return dict(itertools.chain.from_iterable(extracted_versions))


def update_requirement(line, new_versions):
    # convert to pep-508 compatible
    preprocessed = operator_re.sub("==", line)
    requirement = Requirement(preprocessed)

    specifier, *_ = requirement.specifier
    old_version = specifier.version
    new_version = new_versions.get(requirement.name, old_version)

    new_line = f"{requirement.name}={new_version}"

    return new_line


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

    requirements_text = args.requirements.read_text()
    requirements = requirements_text.split("\n")
    new_requirements = [
        update_requirement(line, versions)
        if line and not line.startswith("# ")
        else line
        for line in requirements
    ]
    new_requirements_text = "\n".join(new_requirements)

    if args.dry:
        separator = "\n" + "—" * 80 + "\n"
        print(
            "contents of the old requirements file:",
            requirements_text,
            "contents of the new requirements file:",
            new_requirements_text,
            sep=separator,
            end=separator,
        )
    else:
        args.requirements.write_text(new_requirements_text)
