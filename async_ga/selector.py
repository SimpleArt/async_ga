from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome, Population
import random
from typing import AsyncIterable, AsyncIterator, Generic, List, TypeVar

__all__ = ["Selector", "RandomSelector", "TournamentSelector", "RouletteSelector"]

T = TypeVar("T")


class Selector(ABC, Generic[T]):
    """Abstract base class for creating parent selection methods."""

    @abstractmethod
    async def select(self: Selector[T], population: Population[T]) -> AsyncIterable[Chromosome[T]]:
        """Select some parents from the population."""
        for parent in ():
            yield parent


class RandomSelector(Selector[T], ABC, Generic[T]):
    """Select a parent randomly from the population each iteration based on some weights."""

    async def selector_key(self: RandomSelector[T], weight: float) -> float:
        """A modifiable key function that allows different probabilities that a chromosome is chosen."""
        return weight

    @abstractmethod
    async def get_weights(self: RandomSelector[T], population: Population[T]) -> AsyncIterable[float]:
        """Generate the weights from the population. Low weights -> high probability of becoming a parent."""

    async def select(self: RandomSelector[T], population: Population[T]) -> AsyncIterator[Chromosome[T]]:
        """
        Indefinitely select random parents from the population each iteration based on the weights.

        Sorts the population every iteration.
        """
        population.sort(key=lambda chromosome: chromosome.fitness.mean)
        weights = [weight async for weight in self.get_weights(population)]
        weights.reverse()
        min_weight = min(weights)
        for i, weight in enumerate(weights):
            weights[i] = await self.selector_key(weight - min_weight)
        yield population[random.choices(range(len(population)), weights)[0]]
        async for parent in super().select(population):
            yield parent


class TournamentSelector(RandomSelector[T], Generic[T]):
    """Tournament selection randomly selects chromosomes based on their population ranking."""

    async def selector_key(self: TournamentSelector[T], weight: float) -> float:
        """By default, the tournament selector squares the weights."""
        return weight * weight

    async def get_weights(self: TournamentSelector[T], population: Population[T]) -> AsyncIterator[int]:
        """Generate the weights from the population using the indexes."""
        for i, _ in enumerate(population):
            yield i


class RouletteSelector(RandomSelector[T], Generic[T]):
    """Roulette selection randomly selects chromosomes based on their fitness."""

    async def get_weights(self: RouletteSelector[T], population: Population[T]) -> AsyncIterator[float]:
        """Generate the weights from the population using the fitness means."""
        for chromosome in population:
            yield chromosome.fitness.mean
