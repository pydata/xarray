#!/usr/bin/env python
"""A script that can be quickly run that explores the public API of Xarray
and validates docstrings along the way according to the numpydoc conventions."""

import functools
import importlib
import logging
import sys
import types
from pathlib import Path

from numpydoc.validate import validate

logger = logging.getLogger("numpydoc-public-api")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)

PROJECT_ROOT = (Path(__file__).parent / "..").resolve()
PUBLIC_MODULES = ["xarray"]
ROOT_PACKAGE = "xarray"

# full list of numpydoc error codes: https://numpydoc.readthedocs.io/en/latest/validation.html
SKIP_ERRORS = [ # TODO: Curate these for Xarray
    "GL01",  # parcels is fine with the summary line starting directly after `"""`, or on the next line.
    "SA01",  # Parcels doesn't require the "See also" section
    "SA04",  #
    "ES01",  # We don't require the extended summary for all docstrings
    "EX01",  # We don't require the "Examples" section for all docstrings
    "SS06",  # Not possible to make all summaries one line
    #
    # To be fixed up
    "GL02",  # Closing quotes should be placed in the line after the last text in the docstring (do not close the quotes in the same line as the text, or leave a blank line between the last text and the quotes)
    "GL03",  # Double line break found; please use only one blank line to separate sections or paragraphs, and do not leave blank lines at the end of docstrings
    "GL07",  # Sections are in the wrong order. Correct order is: {correct_sections}
    "GL08",  # The object does not have a docstring
    "SS01",  # No summary found (a short summary in a single line should be present at the beginning of the docstring)
    "SS02",  # Summary does not start with a capital letter
    "SS03",  # Summary does not end with a period
    "SS04",  # Summary contains heading whitespaces
    "SS05",  # Summary must start with infinitive verb, not third person (e.g. use "Generate" instead of "Generates")
    "PR01",  # Parameters {missing_params} not documented
    "PR02",  # Unknown parameters {unknown_params}
    "PR03",  # Wrong parameters order. Actual: {actual_params}. Documented: {documented_params}
    "SA02",  # Missing period at end of description for See Also "{reference_name}" reference
    "SA03",  # Description should be capitalized for See Also
    #
    # TODO consider whether to continue ignoring the following
    "GL09",  # Deprecation warning should precede extended summary
    "GL10",  # reST directives {directives} must be followed by two colons
    "PR04",  # Parameter "{param_name}" has no type
    "PR05",  # Parameter "{param_name}" type should not finish with "."
    "PR06",  # Parameter "{param_name}" type should use "{right_type}" instead of "{wrong_type}"
    "PR07",  # Parameter "{param_name}" has no description
    "PR08",  # Parameter "{param_name}" description should start with a capital letter
    "PR09",  # Parameter "{param_name}" description should finish with "."
    "PR10",  # Parameter "{param_name}" requires a space before the colon separating the parameter name and type
    "RT01",  # No Returns section found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned
    "RT03",  # Return value has no description
    "RT04",  # Return value description should start with a capital letter
    "RT05",  # Return value description should finish with "."
    "YD01",  # No Yields section found
]


def is_built_in(type_or_instance: type | object):
    if isinstance(type_or_instance, type):
        return type_or_instance.__module__ == "builtins"
    else:
        return type_or_instance.__class__.__module__ == "builtins"


def walk_module(module_str: str, public_api: list[str] | None = None) -> list[str]:
    if public_api is None:
        public_api = []

    module = importlib.import_module(module_str)
    try:
        all_ = module.__all__
    except AttributeError:
        print(f"No __all__ variable found in public module {module_str!r}")
        return public_api

    if module_str not in public_api:
        public_api.append(module_str)
    for item_str in all_:
        item = getattr(module, item_str)
        if isinstance(item, types.ModuleType):
            walk_module(f"{module_str}.{item_str}", public_api)
        if isinstance(item, (types.FunctionType,)):
            public_api.append(f"{module_str}.{item_str}")
        elif is_built_in(item):
            print(f"Found builtin at '{module_str}.{item_str}' of type {type(item)}")
            continue
        elif isinstance(item, type):
            public_api.append(f"{module_str}.{item_str}")
            walk_class(module_str, item, public_api)
        else:
            logger.info(
                f"Encountered unexpected public object at '{module_str}.{item_str}' of {item!r} in public API. Don't know how to handle with numpydoc - ignoring."
            )

    return public_api


def get_public_class_attrs(class_: type) -> set[str]:
    return {a for a in dir(class_) if not a.startswith("_")}


def walk_class(module_str: str, class_: type, public_api: list[str]) -> list[str]:
    class_str = class_.__name__

    # attributes that were introduced by this class specifically - not from inheritance
    attrs = get_public_class_attrs(class_) - functools.reduce(
        set.union, (get_public_class_attrs(base) for base in class_.__bases__)
    )

    public_api.extend([f"{module_str}.{class_str}.{attr_str}" for attr_str in attrs])
    return public_api


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate numpydoc docstrings in the public API")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (can be repeated)")
    args = parser.parse_args()

    # Set logging level based on verbosity: 0=WARNING, 1=INFO, 2+=DEBUG
    if args.verbose == 0:
        log_level = logging.WARNING
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG

    logger.setLevel(log_level)

    public_api = []
    for module in PUBLIC_MODULES:
        public_api += walk_module(module)

    errors = 0
    for item in public_api:
        logger.info(f"Processing validating {item}")
        try:
            res = validate(item)
        except (AttributeError, StopIteration, ValueError) as e:
            if isinstance(e, ValueError) and "Error parsing See Also entry" in str(e): # TODO: Fix later https://github.com/pydata/xarray/issues/8596#issuecomment-3832443795
                logger.info(f"Skipping See Also parsing error for {item!r}.")
                continue
            logger.warning(f"Could not process {item!r}. Encountered error. {e!r}")
            continue
        if res["type"] in ("module", "float", "int", "dict"):
            continue
        for err in res["errors"]:
            if err[0] not in SKIP_ERRORS:
                print(f"{item}: {err}")
                errors += 1
    sys.exit(errors)


if __name__ == "__main__":
    main()
