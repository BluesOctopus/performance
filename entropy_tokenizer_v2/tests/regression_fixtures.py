REGRESSION_CASES = [
    {
        "name": "syn_and_plain_code",
        "source": "<SYN_0> foo bar\nfoo = bar\n",
        "rmap": {"foo": "<VAR>", "bar": "<VAR>"},
    },
    {
        "name": "string_and_attribute",
        "source": 'obj.name = "abc"\nprint(obj.name)\n',
        "rmap": {"name": "<ATTR>", '"abc"': "<STR>"},
    },
    {
        "name": "multiline_string",
        "source": 'msg = """hello\nworld"""\nprint(msg)\n',
        "rmap": {"hello": "<VAR>", "msg": "<VAR>"},
    },
    {
        "name": "function_if_return",
        "source": "def f(x):\n    if x:\n        return x\n    return 0\n",
        "rmap": {"x": "<VAR>"},
    },
]
