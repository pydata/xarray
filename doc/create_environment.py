#!/usr/bin/env python3

import pathlib


def main():
    root = pathlib.Path(__file__).parent

    package_root = root.parent / "dist"
    package_paths = sorted(package_root.glob("xarray*.conda"))
    package_path = package_paths[-1]

    template_path = root / "environment_template.yml"
    env_path = root / "environment.yml"

    template = template_path.read_text()
    environment = template.replace('"{{ local-package }}"', str(package_path))

    env_path.write_text(environment)

if __name__ == "__main__":
    main()
