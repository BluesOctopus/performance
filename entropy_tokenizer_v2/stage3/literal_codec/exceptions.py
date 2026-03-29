"""Custom exceptions for literal codec."""


class LiteralCodecError(Exception):
    """Base exception."""


class PrefixConflictError(LiteralCodecError):
    """Raised when a candidate code violates prefix constraints."""


class DecodeError(LiteralCodecError):
    """Raised when an encoded value cannot be decoded uniquely."""
