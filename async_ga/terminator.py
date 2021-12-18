from __future__ import annotations
from abc import ABC, abstractmethod
from time import time
from typing import Any, Generic, TypeVar

__all__ = ["Terminator", "MaxIteration", "TimeLimit"]

T = TypeVar("T")


class Terminator(Generic[T]):
    """Abstract base class for terminating a genetic algorithm."""

    @abstractmethod
    async def is_active(self: Terminator) -> bool:
        """Returns whether the genetic algorithm is still actively running."""
        return True


class MaxIteration(Terminator[T], Generic[T]):
    """Terminates after a maximum number of iterations."""
    iteration: int = 0
    max_iteration: int = 100

    async def is_active(self: MaxIteration[T]) -> bool:
        """Terminates after a maximum number of iterations."""
        return self.iteration < self.max_iteration and await super().is_active()


class TimeLimit(Terminator[T], Generic[T]):
    """Terminates after a maximum time limit."""
    end_time: float
    max_time: float = 60

    def __init__(self: TimeLimit[T], *args: Any, **kwargs: Any) -> None:
        self.end_time = time() + self.max_time
        super().__init__(*args, **kwargs)

    async def is_active(self: Terminator) -> bool:
        """Terminates after a maximum number of iterations."""
        return time() < self.end_time and await super().is_active()
