import sys


def removesuffix(base: str, suffix: str) -> str:
    if sys.version_info >= (3, 9):
        return base.removesuffix(suffix)
    else:
        if base.endswith(suffix):
            return base[: len(base) - len(suffix)]
        return base


def removeprefix(base: str, prefix: str) -> str:
    if sys.version_info >= (3, 9):
        return base.removeprefix(prefix)
    else:
        if base.startswith(prefix):
            return base[len(prefix) :]
        return base
