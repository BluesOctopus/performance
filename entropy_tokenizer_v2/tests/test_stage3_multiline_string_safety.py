from token_scorer import apply_token_replacement


def test_multiline_string_content_is_not_mutated_by_identifier_map() -> None:
    source = 'msg = """hello\nworld"""\nprint(msg)\n'
    out = apply_token_replacement(source, {"hello": "<VAR>"})
    assert out == source
