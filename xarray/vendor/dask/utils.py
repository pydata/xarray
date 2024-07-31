from __future__ import annotations

import functools
import re
from datetime import datetime
from typing import Any


def typename(typ: Any, short: bool = False) -> str:
    """
    Return the name of a type

    Examples
    --------
    >>> typename(int)
    'int'

    >>> from dask.core import literal
    >>> typename(literal)
    'dask.core.literal'
    >>> typename(literal, short=True)
    'dask.literal'
    """
    if not isinstance(typ, type):
        return typename(type(typ))
    try:
        if not typ.__module__ or typ.__module__ == "builtins":
            return typ.__name__
        else:
            if short:
                module, *_ = typ.__module__.split(".")
            else:
                module = typ.__module__
            return module + "." + typ.__name__
    except AttributeError:
        return str(typ)


byte_sizes = {
    "kB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "PB": 10**15,
    "KiB": 2**10,
    "MiB": 2**20,
    "GiB": 2**30,
    "TiB": 2**40,
    "PiB": 2**50,
    "B": 1,
    "": 1,
}
byte_sizes = {k.lower(): v for k, v in byte_sizes.items()}
byte_sizes.update({k[0]: v for k, v in byte_sizes.items() if k and "i" not in k})
byte_sizes.update({k[:-1]: v for k, v in byte_sizes.items() if k and "i" in k})


def format_time(n: float) -> str:
    """format integers as time

    >>> from dask.utils import format_time
    >>> format_time(1)
    '1.00 s'
    >>> format_time(0.001234)
    '1.23 ms'
    >>> format_time(0.00012345)
    '123.45 us'
    >>> format_time(123.456)
    '123.46 s'
    >>> format_time(1234.567)
    '20m 34s'
    >>> format_time(12345.67)
    '3hr 25m'
    >>> format_time(123456.78)
    '34hr 17m'
    >>> format_time(1234567.89)
    '14d 6hr'
    """
    if n > 24 * 60 * 60 * 2:
        d = int(n / 3600 / 24)
        h = int((n - d * 3600 * 24) / 3600)
        return f"{d}d {h}hr"
    if n > 60 * 60 * 2:
        h = int(n / 3600)
        m = int((n - h * 3600) / 60)
        return f"{h}hr {m}m"
    if n > 60 * 10:
        m = int(n / 60)
        s = int(n - m * 60)
        return f"{m}m {s}s"
    if n >= 1:
        return "%.2f s" % n
    if n >= 1e-3:
        return "%.2f ms" % (n * 1e3)
    return "%.2f us" % (n * 1e6)


def format_time_ago(n: datetime) -> str:
    """Calculate a '3 hours ago' type string from a Python datetime.

    Examples
    --------
    >>> from datetime import datetime, timedelta

    >>> now = datetime.now()
    >>> format_time_ago(now)
    'Just now'

    >>> past = datetime.now() - timedelta(minutes=1)
    >>> format_time_ago(past)
    '1 minute ago'

    >>> past = datetime.now() - timedelta(minutes=2)
    >>> format_time_ago(past)
    '2 minutes ago'

    >>> past = datetime.now() - timedelta(hours=1)
    >>> format_time_ago(past)
    '1 hour ago'

    >>> past = datetime.now() - timedelta(hours=6)
    >>> format_time_ago(past)
    '6 hours ago'

    >>> past = datetime.now() - timedelta(days=1)
    >>> format_time_ago(past)
    '1 day ago'

    >>> past = datetime.now() - timedelta(days=5)
    >>> format_time_ago(past)
    '5 days ago'

    >>> past = datetime.now() - timedelta(days=8)
    >>> format_time_ago(past)
    '1 week ago'

    >>> past = datetime.now() - timedelta(days=16)
    >>> format_time_ago(past)
    '2 weeks ago'

    >>> past = datetime.now() - timedelta(days=190)
    >>> format_time_ago(past)
    '6 months ago'

    >>> past = datetime.now() - timedelta(days=800)
    >>> format_time_ago(past)
    '2 years ago'

    """
    units = {
        "years": lambda diff: diff.days / 365,
        "months": lambda diff: diff.days / 30.436875,  # Average days per month
        "weeks": lambda diff: diff.days / 7,
        "days": lambda diff: diff.days,
        "hours": lambda diff: diff.seconds / 3600,
        "minutes": lambda diff: diff.seconds % 3600 / 60,
    }
    diff = datetime.now() - n
    for unit in units:
        dur = int(units[unit](diff))
        if dur > 0:
            if dur == 1:  # De-pluralize
                unit = unit[:-1]
            return f"{dur} {unit} ago"
    return "Just now"


def format_bytes(n: int) -> str:
    """Format bytes as text

    >>> from dask.utils import format_bytes
    >>> format_bytes(1)
    '1 B'
    >>> format_bytes(1234)
    '1.21 kiB'
    >>> format_bytes(12345678)
    '11.77 MiB'
    >>> format_bytes(1234567890)
    '1.15 GiB'
    >>> format_bytes(1234567890000)
    '1.12 TiB'
    >>> format_bytes(1234567890000000)
    '1.10 PiB'

    For all values < 2**60, the output is always <= 10 characters.
    """
    for prefix, k in (
        ("Pi", 2**50),
        ("Ti", 2**40),
        ("Gi", 2**30),
        ("Mi", 2**20),
        ("ki", 2**10),
    ):
        if n >= k * 0.9:
            return f"{n / k:.2f} {prefix}B"
    return f"{n} B"


hex_pattern = re.compile("[a-f]+")


@functools.lru_cache(100000)
def key_split(s):
    """
    >>> key_split("x")
    'x'
    >>> key_split("x-1")
    'x'
    >>> key_split("x-1-2-3")
    'x'
    >>> key_split(("x-2", 1))
    'x'
    >>> key_split("('x-2', 1)")
    'x'
    >>> key_split("('x', 1)")
    'x'
    >>> key_split("hello-world-1")
    'hello-world'
    >>> key_split(b"hello-world-1")
    'hello-world'
    >>> key_split("ae05086432ca935f6eba409a8ecd4896")
    'data'
    >>> key_split("<module.submodule.myclass object at 0xdaf372")
    'myclass'
    >>> key_split(None)
    'Other'
    >>> key_split("x-abcdefab")  # ignores hex
    'x'
    >>> key_split("_(x)")  # strips unpleasant characters
    'x'
    """
    if type(s) is bytes:
        s = s.decode()
    if type(s) is tuple:
        s = s[0]
    try:
        words = s.split("-")
        if not words[0][0].isalpha():
            result = words[0].split(",")[0].strip("_'()\"")
        else:
            result = words[0]
        for word in words[1:]:
            if word.isalpha() and not (
                len(word) == 8 and hex_pattern.match(word) is not None
            ):
                result += "-" + word
            else:
                break
        if len(result) == 32 and re.match(r"[a-f0-9]{32}", result):
            return "data"
        else:
            if result[0] == "<":
                result = result.strip("<>").split()[0].split(".")[-1]
            return result
    except Exception:
        return "Other"
