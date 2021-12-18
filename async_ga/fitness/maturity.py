from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome
import asyncio
from math import sqrt
from time import time
from typing import Generic, TypeVar

__all__ = ["Maturity", "AgeMaturity", "DeterministicMaturity", "StatisticsMaturity", "TimeMaturity", "VarianceMaturity"]

T = TypeVar("T")


class Maturity(ABC, Generic[T]):
    """Abstract base class for checking if a chromosome is mature."""

    @abstractmethod
    async def is_mature(self: Maturity[T], chromosome: Chromosome[T]) -> bool:
        """
        Returns if the chromosome is mature.

        Subclasses should call `await super().is_mature(chromosome)` to ensure
        other maturity checks are checked.
        """
        return hasattr(chromosome.fitness, "mean") and hasattr(chromosome.fitness, "variance")


class AgeMaturity(Maturity[T], Generic[T]):
    """Checks if a chromosome is mature based on its age."""
    age: int = 10
    iteration: int = 0

    async def is_mature(self: AgeMaturity[int], chromosome: Chromosome[T]) -> bool:
        """Checks if a chromosome is mature based on its age, plus the square root of the current iteration."""
        return chromosome.age >= self.age + sqrt(self.iteration) and await super().is_mature(chromosome)


class DeterministicMaturity(Maturity[T], Generic[T]):
    """For deterministic fitnesses, wait until the chromosome is fully matured."""

    async def is_mature(self: AgeMaturity[int], chromosome: Chromosome[T]) -> bool:
        """Wait until the chromosome is fully matured."""
        return not chromosome.is_active and await super().is_mature(chromosome)


class StatisticsMaturity(Maturity[T], Generic[T]):
    """Checks if a chromosome is mature based on its test statistic."""
    test_statistic: float = 1.0

    async def is_mature(self: StatisticsMaturity[int], chromosome: Chromosome[T]) -> bool:
        """Checks if a chromosome is mature based on its fitness variance."""
        return (
            chromosome.age > 1
            and sqrt(chromosome.fitness.variance / (chromosome.age - 1)) < self.test_statistic
            and await super().is_mature(chromosome)
        )


class TimeMaturity(Maturity[T], Generic[T]):
    """Checks if a chromosome is mature based on how much time has elapsed."""
    timeout: float = 0.1

    async def is_mature(self: TimeMaturity[int], chromosome: Chromosome[T]) -> bool:
        """Checks if a chromosome is mature based on how much time has elapsed."""
        start = time()
        if not await super().is_mature(chromosome):
            return False
        await asyncio.sleep(start - time() + self.timeout)
        return True


class VarianceMaturity(Maturity[T], Generic[T]):
    """Checks if a chromosome is mature based on its fitness variance."""
    variance: float = 0.1

    async def is_mature(self: VarianceMaturity[int], chromosome: Chromosome[T]) -> bool:
        """Checks if a chromosome is mature based on its fitness variance."""
        return chromosome.fitness < self.variance and await super().is_mature(chromosome)
