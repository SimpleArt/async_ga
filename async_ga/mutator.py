from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome
import random
from typing import AsyncIterable, AsyncIterator, Generic, Iterable, TypeVar, Union

__all__ = ["Mutator", "GeneMutator", "NoiseMutator", "IntMutator", "SwapMutator"]

T = TypeVar("T")


class Mutator(ABC, Generic[T]):
    """Abstract base class for chromosome mutation methods."""

    @abstractmethod
    async def mutate(self: Mutator[T], chromosome: Chromosome[T]) -> AsyncIterable[Union[AsyncIterable[T], Iterable[T]]]:
        """Generate mutated versions of a chromosome."""
        for mutation in ():
            yield mutation


class GeneMutator(Mutator[T], Generic[T]):
    """Replaces random genes with new random genes."""
    repeat: int = 1

    @abstractmethod
    async def initial_gene(self: Mutator[T], /) -> T:
        """Generate a random gene value."""

    async def mutate(self: Mutator[T], chromosome: Chromosome[T]) -> AsyncIterator[Chromosome[T]]:
        """Replaces random genes with new random genes."""
        for _ in range(self.repeat):
            chromosome[random.randrange(len(chromosome))] = await self.initial_gene()
            yield chromosome
        async for mutation in super().mutate(chromosome):
            yield mutation


class NoiseMutator(Mutator[float]):
    """Add gaussian noise to random genes."""
    repeat: int = 1
    deviation: float = 1.0

    async def mutate(self: NoiseMutator, chromosome: Chromosome[float]) -> AsyncIterator[Chromosome[float]]:
        """Add gaussian noise to random genes."""
        for _ in range(self.repeat):
            chromosome[random.randrange(len(chromosome))] += random.gauss(0, self.deviation)
            yield chromosome
        async for mutation in super().mutate(chromosome):
            yield mutation


class IntMutator(Mutator[int]):
    """Increment or decrement random genes."""
    repeat: int = 1

    async def mutate(self: IntMutator, chromosome: Chromosome[int]) -> AsyncIterator[Chromosome[int]]:
        """Increment or decrement random genes."""
        for _ in range(self.repeat):
            chromosome[random.randrange(len(chromosome))] += random.choice([-1, 1])
            yield chromosome
        async for mutation in super().mutate(chromosome):
            yield mutation


class SwapMutator(Mutator[int]):
    """Randomly swap genes."""
    repeat: int = 1

    async def mutate(self: IntMutator, chromosome: Chromosome[int]) -> AsyncIterator[Chromosome[int]]:
        """Randomly swap genes."""
        for _ in range(self.repeat):
            i1, i2 = random.sample(range(len(chromosome)), k=2)
            chromosome[i1], chromosome[i2] = chromosome[i2], chromosome[i1]
            yield chromosome
        async for mutation in super().mutate(chromosome):
            yield mutation
