from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome
from typing import AsyncIterable, Generic, TypeVar

__all__ = ["FitnessFunction"]

T = TypeVar("T")


class FitnessFunction(Generic[T]):
    """Abstract base class for computing fitness estimates of chromosomes."""

    @abstractmethod
    async def fitness_of(self: FitnessFunction[T], chromosome: Chromosome[T]) -> AsyncIterable[float]:
        """Generates estimates of the chromosome's fitness."""
        for fitness in ():
            yield fitness
