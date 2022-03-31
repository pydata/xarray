from datatree.utils import removeprefix, removesuffix


def checkequal(expected_result, obj, method, *args, **kwargs):
    result = method(obj, *args, **kwargs)
    assert result == expected_result


def checkraises(exc, obj, method, *args):
    try:
        method(obj, *args)
    except Exception as e:
        assert isinstance(e, exc) is True


def test_removeprefix():
    checkequal("am", "spam", removeprefix, "sp")
    checkequal("spamspam", "spamspamspam", removeprefix, "spam")
    checkequal("spam", "spam", removeprefix, "python")
    checkequal("spam", "spam", removeprefix, "spider")
    checkequal("spam", "spam", removeprefix, "spam and eggs")
    checkequal("", "", removeprefix, "")
    checkequal("", "", removeprefix, "abcde")
    checkequal("abcde", "abcde", removeprefix, "")
    checkequal("", "abcde", removeprefix, "abcde")

    checkraises(TypeError, "hello", removeprefix)
    checkraises(TypeError, "hello", removeprefix, 42)
    checkraises(TypeError, "hello", removeprefix, 42, "h")
    checkraises(TypeError, "hello", removeprefix, "h", 42)
    checkraises(TypeError, "hello", removeprefix, ("he", "l"))


def test_removesuffix():
    checkequal("sp", "spam", removesuffix, "am")
    checkequal("spamspam", "spamspamspam", removesuffix, "spam")
    checkequal("spam", "spam", removesuffix, "python")
    checkequal("spam", "spam", removesuffix, "blam")
    checkequal("spam", "spam", removesuffix, "eggs and spam")

    checkequal("", "", removesuffix, "")
    checkequal("", "", removesuffix, "abcde")
    checkequal("abcde", "abcde", removesuffix, "")
    checkequal("", "abcde", removesuffix, "abcde")

    checkraises(TypeError, "hello", removesuffix)
    checkraises(TypeError, "hello", removesuffix, 42)
    checkraises(TypeError, "hello", removesuffix, 42, "h")
    checkraises(TypeError, "hello", removesuffix, "h", 42)
    checkraises(TypeError, "hello", removesuffix, ("lo", "l"))
