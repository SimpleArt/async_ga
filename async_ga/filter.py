from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome, Population
import random
from typing import AsyncIterable, AsyncIterator, Generic, List, TypeVar, Union

__all__ = ["Filter", "EliteFilter", "RandomFilter", "TournamentFilter", "RouletteFilter"]

T = TypeVar("T")


class Filter(ABC, Generic[T]):
    """Abstract base class for filtering out chromosomes from the population."""

    @abstractmethod
    async def filter_nonsurvivors(self: Filter[T], population: Population[T]) -> AsyncIterable[Chromosome[T]]:
        """Remove and yield chromosome(s) from the population."""
        for chromosome in ():
            yield


class EliteFilter(Filter[T], Generic[T]):
    """Filter chromosomes using elitism, removing chromosomes which have bad fitnesses."""
    population_length: int

    async def filter_nonsurvivors(self: EliteFilter[T], population: Population[T]) -> AsyncIterator[Chromosome[T]]:
        """Filter chromosomes using elitism, removing chromosomes which have bad fitnesses."""
        if len(population) > self.population_length:
            yield population.pop()
        async for chromosome in super().filter_nonsurvivors(population):
            yield chromosome


class RandomFilter(Filter[T], ABC, Generic[T]):
    """Remove a random chromosome based on some weights."""
    population_length: int

    async def filter_key(self: RandomFilter[T], weight: float) -> float:
        """A modifiable key function that allows different probabilities that a chromosome is chosen."""
        return weight

    @abstractmethod
    async def get_weights(self: RandomFilter[T], population: Population[T]) -> AsyncIterable[float]:
        """Generate the weights from the population. Low weights -> low probability of being removed."""

    async def filter_nonsurvivors(self: RandomFilter[T], population: Population[T]) -> AsyncIterator[Chromosome[T]]:
        """Remove a random chromosome based on some weights."""
        if len(population) > self.population_length:
            weights = [weight async for weight in self.get_weights(population)]
            min_weight = min(weights)
            for i, weight in enumerate(weights):
                weights[i] = await self.filter_key(weight - min_weight)
            yield population.pop(random.choices(range(len(population)), weights)[0])
        async for chromosome in super().filter_nonsurvivors(population, chromosome):
            yield chromosome


class TournamentFilter(RandomFilter[T], Generic[T]):
    """Tournament filtering randomly selects chromosomes based on their population ranking."""

    async def filter_key(self: TournamentFilter[T], weight: float) -> float:
        """By default, the tournament filter squares the weights."""
        return weight * weight

    async def get_weights(self: TournamentFilter[T], population: Population[T]) -> AsyncIterator[int]:
        """Generate the weights from the population using the indexes."""
        for i, _ in enumerate(population):
            yield i


class RouletteFilter(RandomFilter[T], Generic[T]):
    """Roulette filtering randomly selects chromosomes based on their fitness means."""

    async def get_weights(self: RouletteFilter[T], population: Population[T]) -> AsyncIterator[float]:
        """Generate the weights from the population using the fitness means."""
        for chromosome in population:
            yield chromosome.fitness.mean
