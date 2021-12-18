from __future__ import annotations
from collections import UserList
import sys
from typing import Any, Generic, Iterable, List, MutableSequence, Optional, TypeVar

__all__ = ["Chromosome", "Fitness", "Population"]

T = TypeVar("T")

if sys.version_info < (3, 9):
    class UserList(UserList, MutableSequence[T], Generic[T]):
        data: List[T]


class Fitness:
    """Stores the fitness statistics."""
    mean: float
    variance: float


class Chromosome(UserList[T], Generic[T]):
    """
    Create a chromosome using:
        await ga.create_chromosome(data)

    Chromosomes behave like normal lists, but they also store their
    age, fitness statistics, and whether or not the chromosome is
    still actively updating its fitness.
    """
    age: int
    fitness: Fitness
    is_active: bool

    def __init__(self: Chromosome[T], data: Iterable[T] = (), /) -> None:
        self.age = 0
        self.data = list(data)
        self.fitness = Fitness()
        self.is_active = True


Population = List[Chromosome[T]]
