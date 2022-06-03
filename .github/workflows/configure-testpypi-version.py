import argparse
import copy
import pathlib

import tomli
import tomli_w


def split_path(path):
    if isinstance(path, str):
        return [part for part in path.split("/") if part]
    else:
        return path


def extract(mapping, path):
    parts = split_path(path)
    cur = mapping
    for part in parts:
        cur = cur[part]

    return cur


def update(mapping, path, value):
    new = copy.deepcopy(mapping)

    parts = split_path(path)
    parent = extract(new, parts[:-1])
    parent[parts[-1]] = value

    return new


parser = argparse.ArgumentParser()
parser.add_argument("path", type=pathlib.Path)
args = parser.parse_args()

content = args.path.read_text()
decoded = tomli.loads(content)
updated = update(decoded, "tool/setuptools_scm/local_scheme", "no-local-version")

new_content = tomli_w.dumps(updated)

args.path.write_text(new_content)
