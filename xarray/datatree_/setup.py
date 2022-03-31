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
    name="xarray-datatree",
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
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    python_requires=">=3.8",
    use_scm_version={"version_scheme": "post-release", "local_scheme": "dirty-tag"},
    setup_requires=["setuptools_scm>=3.4", "setuptools>=42"],
)
