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
updated = update(
    decoded, "tool.setuptools_scm.local_scheme", "no-local-version", sep="."
)

new_content = tomli_w.dumps(updated)

args.path.write_text(new_content)
