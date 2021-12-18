from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome
import random
from typing import AsyncIterable, AsyncIterator, Generic, Iterable, Iterator, TypeVar, Union

__all__ = ["Crosser", "UniformCrosser", "SinglePointCrosser", "ArithmeticCrosser"]

Numeric = TypeVar("Numeric", bound=float)
T = TypeVar("T")


class Crosser(ABC, Generic[T]):
    """Abstract base class for parent crossover methods."""

    @abstractmethod
    async def cross(self: Crosser[T], parent1: Chromosome[T], parent2: Chromosome[T], /) -> AsyncIterable[Union[AsyncIterable[T], Iterable[T]]]:
        """Cross two parents and produce children from them."""
        for child in ():
            yield


class UniformCrosser(Crosser[T], Generic[T]):
    """Cross parents by choosing genes from each parent randomly."""
    repeat: int = 1

    async def cross(self: UniformCrosser[T], parent1: Chromosome[T], parent2: Chromosome[T], /) -> AsyncIterator[Iterator[T]]:
        """Cross two parents, choosing each gene randomly."""
        for _ in range(self.repeat):
            yield map(self.random_gene, parent1, parent2)
        async for child in super().cross(parent1, parent2):
            yield child

    @staticmethod
    def random_gene(*genes: T) -> T:
        return random.choice(genes)


class SinglePointCrosser(Crosser[T], Generic[T]):
    """Cross parents by choosing genes from one parent up to a crossover point, and then swapping to the second parent."""
    repeat: int = 1

    async def cross(self: SinglePointCrosser[T], parent1: Chromosome[T], parent2: Chromosome[T], /) -> AsyncIterator[Chromosome[T]]:
        """Cross two parents and produce 2 varying children from them."""
        for _ in range(self.repeat):
            i = random.randrange(1, len(parent1))
            yield parent1[:i] + parent2[i:]
            yield parent2[:i] + parent1[i:]
        async for child in super().cross(parent1, parent2):
            yield child


class ArithmeticCrosser(Crosser[Numeric], Generic[Numeric]):
    """Cross parents by averaging their genes."""

    async def cross(self: SinglePointCrosser[Numeric], parent1: Chromosome[Numeric], parent2: Chromosome[Numeric], /) -> AsyncIterator[Iterator[Numeric]]:
        """Cross two parents and produce 2 varying children from them."""
        # Take the averages of the parents' genes.
        yield map(self.average, parent1, parent2)
        async for child in super().cross(parent1, parent2):
            yield child

    @staticmethod
    def average(gene1: Numeric, gene2: Numeric) -> Numeric:
        """Returns the average of two genes, rounding (up/down randomly) integer genes."""
        if isinstance(gene1, int) and isinstance(gene2, int):
            return round((gene1 + gene2 - 1) / 2 + random.random())
        else:
            return (gene1 + gene2) / 2
