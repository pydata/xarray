from os.path import exists

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

if exists("README.rst"):
    with open("README.rst") as f:
        long_description = f.read()
else:
    long_description = ""


setup(
    name="datatree",
    version="0.0.1",
    description="Hierarchical tree-like data structures for xarray",
    long_description=long_description,
    url="https://github.com/xarray-contrib/datatree",
    author="Thomas Nicholas",
    author_email="thomas.nicholas@columbia.edu",
    license="Apache",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    python_requires=">=3.9",
    setup_requires="setuptools_scm",
    use_scm_version={
        "write_to": "datatree/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
