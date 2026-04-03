from __future__ import annotations


def test_stage3_imports_without_sys_path_hacks() -> None:
    import stage3.backends  # noqa: F401
    import stage3.routing.router  # noqa: F401
    import stage3.routing.rules  # noqa: F401
    import stage3.exact.alias_codec  # noqa: F401
    import stage3.lexical.semantic_codec  # noqa: F401
    import stage3.lexical.string_classifier  # noqa: F401

