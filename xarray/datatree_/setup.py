from setuptools import find_packages, setup

install_requires = [
    "xarray>=0.19.0",
    "netcdf4"
    "anytree",
    "future",
]

extras_require = {'tests':
    [
        "pytest",
        "flake8",
        "black",
        "codecov",
    ]
}

setup(
    name="datatree",
    description="Hierarchical tree-like data structures for xarray",
    url="https://github.com/TomNicholas/datatree",
    author="Thomas Nicholas",
    author_email="thomas.nicholas@columbia.edu",
    license="Apache",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["docs", "tests", "tests.*", "docs.*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.7",
    setup_requires="setuptools_scm",
    use_scm_version={
        "write_to": "datatree/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
)
