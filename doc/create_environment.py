#!/usr/bin/env python3

import pathlib


def main():
    root = pathlib.Path(__file__).parent

    local_channel = root.parent / "local_channel"
    package_paths = list(local_channel.joinpath("noarch").glob("xarray*.conda"))
    if len(package_paths) != 1:
        raise RuntimeError(f"zero or more than one package found: {package_paths}")

    package_path = package_paths[0]

    template_path = root / "environment_template.yml"
    env_path = root / "environment.yml"

    template = template_path.read_text()
    environment = template.replace('"{{ local-package }}"', str(package_path))

    env_path.write_text(environment)

if __name__ == "__main__":
    main()
