# -*- coding: utf-8 -*-
"""
src/core/result.py

Defines the project-wide ``Result`` type.

Every provider module in this project imports ``Result`` from this location:

    from src.core.result import Result

``Result`` is a generic container that encodes either a successful value
(``Result.Ok``) or a failure (``Result.Err``) without raising exceptions.
The API is intentionally minimal so every module across the pipeline can use
it consistently.

Usage:

    from src.core.result import Result

    def do_something() -> Result:
        try:
            value = expensive_operation()
            return Result.Ok(value)
        except Exception as exc:
            return Result.Err(exc)

    r = do_something()
    if r.is_ok:
        print(r.unwrap())
    else:
        print(f"Error: {r.error}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


@dataclass
class Result(Generic[T]):
    """Generic result container.

    Parameters
    ----------
    value:
        The wrapped success value.  ``None`` when the result represents an error.
    error:
        The wrapped exception.  ``None`` when the result represents a success.

    Do not instantiate directly; use the class-method factories ``Ok`` and
    ``Err`` instead.
    """

    value: T | None = field(default=None)
    error: Exception | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_ok(self) -> bool:
        """Return ``True`` when the result holds a successful value."""
        return self.error is None

    @property
    def is_err(self) -> bool:
        """Return ``True`` when the result holds an exception."""
        return self.error is not None

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------

    def unwrap(self) -> T:
        """Return the value or raise the stored exception.

        Raises
        ------
        Exception
            The stored exception when ``is_err`` is ``True``.
        """
        if self.is_err:
            raise self.error  # type: ignore[misc]
        return self.value  # type: ignore[return-value]

    def unwrap_or(self, default: T) -> T:
        """Return the value or *default* when the result is an error."""
        if self.is_err:
            return default
        return self.value  # type: ignore[return-value]

    def unwrap_err(self) -> Exception:
        """Return the stored exception or raise ``ValueError`` if there is none."""
        if self.is_ok:
            raise ValueError("Called unwrap_err on an Ok result")
        return self.error  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def Ok(cls, value: T) -> "Result[T]":  # noqa: N802
        """Create a successful result wrapping *value*."""
        return cls(value=value, error=None)

    @classmethod
    def Err(cls, error: Exception) -> "Result":  # noqa: N802
        """Create an error result wrapping *error*."""
        return cls(value=None, error=error)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        return self.is_ok

    def __repr__(self) -> str:
        if self.is_ok:
            return f"Result.Ok({self.value!r})"
        return f"Result.Err({self.error!r})"


__all__ = ["Result"]
