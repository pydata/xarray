import argparse
import copy
import pathlib

import tomli
import tomli_w


def split_path(path, sep="/"):
    if isinstance(path, str):
        return [part for part in path.split(sep) if part]
    else:
        return path


def extract(mapping, path, sep="/"):
    parts = split_path(path, sep=sep)
    cur = mapping
    for part in parts:
        cur = cur[part]

    return cur


def update(mapping, path, value, sep="/"):
    new = copy.deepcopy(mapping)

    parts = split_path(path, sep=sep)
    parent = extract(new, parts[:-1])
    parent[parts[-1]] = value

    return new


parser = argparse.ArgumentParser()
parser.add_argument("path", type=pathlib.Path)
args = parser.parse_args()

content = args.path.read_text()
decoded = tomli.loads(content)
with_local_scheme = update(
    decoded, "tool.setuptools_scm.local_scheme", "no-local-version", sep="."
)
# work around a bug in setuptools / setuptools-scm
with_setuptools_pin = copy.deepcopy(with_local_scheme)
requires = extract(with_setuptools_pin, "build-system.requires", sep=".")
requires[0] = "setuptools>=42,<60"

new_content = tomli_w.dumps(with_setuptools_pin)
args.path.write_text(new_content)
