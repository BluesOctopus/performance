from literal_codec.codebook.decoder import FieldDecoder
from literal_codec.codebook.encoder import FieldEncoder
from literal_codec.types import CodeAssignment, FieldCodebook


def test_roundtrip_and_escape():
    book = FieldCodebook(
        field_name="env",
        version="v1",
        escape_prefix="__L__",
        assignments=[
            CodeAssignment("prod", "a", 1, 1, 0.2),
            CodeAssignment("test", "b", 1, 1, 0.1),
        ],
        metadata={},
    )
    encoder = FieldEncoder(book)
    decoder = FieldDecoder(book)

    assert decoder.decode_value(encoder.encode_value("prod")) == "prod"
    escaped = encoder.encode_value("__L__already_prefixed")
    assert escaped.startswith("__L____L__")
    assert decoder.decode_value(escaped) == "__L__already_prefixed"
