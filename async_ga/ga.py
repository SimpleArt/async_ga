from __future__ import annotations
from abc import ABC, abstractmethod
from async_ga.chromosome import Chromosome, Population
from async_ga.crosser import Crosser, SinglePointCrosser
from async_ga.filter import Filter, EliteFilter
from async_ga.fitness.function import FitnessFunction
from async_ga.fitness.maturity import Maturity, AgeMaturity, StatisticsMaturity
from async_ga.mutator import Mutator, NoiseMutator
from async_ga.selector import Selector, TournamentSelector
from async_ga.terminator import Terminator, MaxIteration
import asyncio
from collections import deque
import random
from typing import Any, AsyncIterable, AsyncIterator, Awaitable, Deque, Iterable, Generic, Literal, List, Optional, Set, Tuple, TypeVar, Union

__all__ = ["AsyncGA", "DefaultGA"]

T = TypeVar("T")

async def pairwise(iterable: AsyncIterable[T]) -> AsyncIterator[Tuple[T, T]]:
    """Groups consecutive pairs of items."""
    MISSING_ITEM = object()
    x1 = MISSING_ITEM
    async for x2 in iterable:
        if x1 is not MISSING_ITEM:
            yield x1, x2
        x1 = x2


class AsyncGA(Crosser[T], Filter[T], FitnessFunction[T], Maturity[T], Mutator[T], Selector[T], Terminator[T], ABC, Generic[T]):
    """
    The base asynchronous genetic algorithm class.

    Build a custom genetic algorithm class using multi-inheritance or by
    implementing the required abstract methods.

    See `async_ga.DefaultGA` for a simple example.

    Run a custom genetic algorithm class using either:
        In a synchronous context:
            population = ga.run()
        In an asynchronous context:
            population = await ga.main()
        Looping in an asynchronous context:
            async for population in ga.evolve():
                ...

    Note:
        The genetic algorithm will attempt to minimize the chromosome fitness.
    """
    _tasks: Set[Awaitable[Any]]
    chromosome_length: int
    fitness_mode: Literal["moving", "square"] = "moving"
    iteration: int = 0
    population_length: int

    async def initial_gene(self: AsyncGA[T], /) -> T:
        """Generate a random gene value."""
        raise NotImplementedError("No random initial gene is given.")

    async def initial_chromosome(self: AsyncGA[T], /) -> AsyncIterable[T]:
        """Default chromosome generator using random genes of the provided chromosome size."""
        for _ in range(self.chromosome_length):
            yield await self.initial_gene()
            await asyncio.sleep(0)

    async def initial_population(self: AsyncGA[T], /) -> AsyncIterable[Union[AsyncIterable[T], Iterable[T]]]:
        """Default population generator using random chromosomes of the provided population size."""
        for _ in range(self.population_length):
            yield self.initial_chromosome()
            await asyncio.sleep(0)

    async def setup(self: AsyncGA[T], population: Population[T]) -> None:
        """Any additional setup required for the population."""
        pass

    async def _update_fitness_task(self: AsyncGA[T], chromosome: Chromosome[T], /) -> None:
        """
        Update a chromosome's fitness in the background based on the `fitness_mode`.

        self.fitness_mode
        ------------------
            "moving", default:
                Uses an exponential moving average of the fitness mean and variance.
                The result is biased towards the last fitness estimate.
                Avoids over-/under-flow, and does not require saving previous iterations.
            "square":
                Uses an arithmetic average of the fitness mean and variance.
                The result is based on every fitness estimate equally.
                Avoids over-/under-flow, and does not require saving previous iterations.
        """
        try:
            if self.fitness_mode == "moving":
                weight = 0.0
                mean = 0.0
                variance = 0.0
                iterator = self.fitness_of(chromosome).__aiter__()
                async for fitness in iterator:
                    weight += 0.01 * (1 - weight)
                    mean += 0.01 * (fitness - mean)
                    variance += 0.01 * (mean / weight - variance)
                    chromosome.age += 1
                    chromosome.fitness.mean = mean / weight
                    chromosome.fitness.variance = variance / weight
                    await asyncio.sleep(0)
                    if not chromosome.is_active:
                        if hasattr(iterator, "aclose"):
                            await iterator.aclose()
                        break
            elif self.fitness_mode == "square":
                n = 0
                mean = 0.0
                mean_squares = 0.0
                iterator = self.fitness_of(chromosome).__aiter__()
                async for fitness in iterator:
                    n += 1
                    mean += (fitness - mean) / n
                    mean_squares += (fitness ** 2 - mean_squares) / n
                    chromosome.age += 1
                    chromosome.fitness.mean = mean
                    chromosome.fitness.variance = mean_squares - mean ** 2
                    await asyncio.sleep(0)
                    if not chromosome.is_active:
                        if hasattr(iterator, "aclose"):
                            await iterator.aclose()
                        break
            else:
                raise ValueError(f"unknown fitness mode {self.fitness_mode!r}")
        finally:
            chromosome.is_active = False

    async def create_chromosome(self: AsyncGA[T], data: Union[AsyncIterable[T], Iterable[T]], /) -> Chromosome[T]:
        """Create a chromosome using the provided data."""
        # Setup the chromosome.
        chromosome = Chromosome()
        chromosome.data = [gene async for gene in data] if isinstance(data, AsyncIterable) else list(data)
        # Start evaluating the chromosome's fitness.
        task = asyncio.create_task(self._update_fitness_task(chromosome))
        self._tasks.add(task)
        while True:
            # Stop once the chromosome is sufficiently mature.
            if await self.is_mature(chromosome):
                return chromosome
            # Stop looping once the fitness is no longer being updated.
            elif task.done():
                break
            # Give time for the chromosome to mature.
            await asyncio.sleep(0)
        # Check for potential errors while updating the fitness.
        await task.result()
        # Free up the reference to the fitness updating task.
        del self._tasks[task]
        # If the chromosome is at least usable, return it.
        if hasattr(chromosome.fitness, "mean") and hasattr(chromosome.fitness, "variance"):
            return chromosome
        # Raise ValueError if the chromosome is not usable.
        else:
            raise AttributeError("chromosome.fitness.mean and chromosome.fitness.variance were never set")

    async def _generate_parents(self: AsyncGA[T], population: Population[T]) -> AsyncIterable[Chromosome[T]]:
        """Generates parents from the population using `self.select` while `self.is_active()`."""
        if not await self.is_active():
            return
        while True:
            iterator = self.select(population).__aiter__()
            async for parent in iterator:
                yield parent
                if not await self.is_active():
                    if hasattr(iterator, "aclose"):
                        await iterator.aclose()
                    return
                await asyncio.sleep(0)

    @staticmethod
    async def _to_buffer(awaitable: Awaitable[T], buffer: Deque[T]) -> None:
        buffer.append(await awaitable)

    async def _generate_children(self: AsyncGA[T], parent1: Chromosome[T], parent2: Chromosome[T], /) -> AsyncIterable[Chromosome[T]]:
        """Generates children from the parents using `self.cross` and packages the data into chromosomes."""
        tasks = []
        buffer = deque()
        async for child in self.cross(parent1, parent2):
            tasks.append(asyncio.create_task(self._to_buffer(self.create_chromosome(child), buffer)))
            while len(buffer) > 0:
                yield buffer.popleft()
                await asyncio.sleep(0)
            await asyncio.sleep(0)
        for task in tasks:
            await task
            while len(buffer) > 0:
                yield buffer.popleft()
                await asyncio.sleep(0)
            await asyncio.sleep(0)

    async def _generate_mutations(self: AsyncGA[T], child: Chromosome[T], /) -> AsyncIterable[Chromosome[T]]:
        """Generates mutated children from the child using `self.mutate` and packages the data into chromosomes."""
        tasks = []
        buffer = deque()
        async for mutation in self.mutate(child):
            tasks.append(asyncio.create_task(self._to_buffer(self.create_chromosome(mutation), buffer)))
            while len(buffer) > 0:
                yield buffer.popleft()
                await asyncio.sleep(0)
            await asyncio.sleep(0)
        for task in tasks:
            await task
            while len(buffer) > 0:
                yield buffer.popleft()
                await asyncio.sleep(0)
            await asyncio.sleep(0)

    async def evolve(self: AsyncGA[T], /) -> AsyncIterator[Population[T]]:
        self._tasks = set()
        try:
            # Get the starting population.
            population = [
                await self.create_chromosome(chromosome)
                async for chromosome in self.initial_population()
            ]
            population.sort(key=lambda chromosome: chromosome.fitness.mean)
            await self.setup(population)
            yield population
            async for parent1, parent2 in pairwise(self._generate_parents(population)):
                self.iteration += 1
                # Add to the population.
                async for child in self._generate_children(parent1, parent2):
                    child.is_active = False
                    population.extend([
                        mutation
                        async for mutation in self._generate_mutations(child)
                    ])
                population.sort(key=lambda chromosome: chromosome.fitness.mean)
                # Kill those that couldn't survive and stop computing their fitnesses.
                async for chromosome in self.filter_nonsurvivors(population):
                    chromosome.is_active = False
                yield population
        finally:
            # Finish computing the fitnesses.
            for chromosome in population:
                chromosome.is_active = False
            await asyncio.gather(*self._tasks)
            self._tasks.clear()

    async def main(self: AsyncGA[T], /) -> Population[T]:
        """Returns the last population from `self.evolve()`."""
        iterator = self.evolve().__aiter__()
        try:
            async for population in iterator:
                pass
            return population
        finally:
            iterator.aclose()

    def run(self: AsyncGA[T], /) -> Population[T]:
        """Shortcut for `asyncio.run(self.main())`."""
        return asyncio.run(self.main())


class DefaultGA(
    SinglePointCrosser[float],
    EliteFilter[float],
    AgeMaturity[float],
    StatisticsMaturity[float],
    NoiseMutator,
    TournamentSelector[float],
    MaxIteration[float],
    AsyncGA[float],
):
    """
    A genetic algorithm with default settings.

    Run it using:
        >>> ga = DefaultGA()
        >>> population = ga.run()
        >>> print(population[0])
        >>> print(population[0].fitness.mean)

    For more advanced running:
        >>> async def main():
        ...     ga = DefaultGA()
        ...     async for population in ga.evolve():
        ...         print(population[0].fitness.mean)
        ...
        >>> asyncio.run(main())

    Default Methods
    ----------------
        Crosser:
            SinglePointCrosser
        Filter:
            EliteFilter
        FitnessFunction:
            Minimize the distance to 5.
        Maturity:
            AgeMaturity, minimum 10 iterations.
            StatisticsMaturity, maximum 0.5 test statistic.
        Mutator:
            NoiseMutator
        Selector:
            TournamentSelector
        Terminator:
            MaxIteration, maximum 50 iterations.

    Default Attributes
    -------------------
        chromosome_length:
            10
        max_iteration:
            50
        min_age:
            5
        percent:
            0.75
        population_length:
            25
        test_statistic:
            0.5
    """
    chromosome_length = 10
    max_iteration = 50
    min_age = 10
    percent: float = 0.75
    population_length = 10
    test_statistic = 0.1

    async def initial_gene(self: DefaultGA) -> float:
        """Create a random gene from 0 to 10."""
        return 10 * random.random()

    async def fitness_of(self: DefaultGA, chromosome: Chromosome[int]) -> AsyncIterator[float]:
        """Generates 0 if a gene is 5 and 1 if a gene is not 5."""
        while True:
            # Only test the genes of a random percent of the chromosome each iteration.
            yield sum(
                (gene - 5) ** 2
                for gene in random.choices(chromosome, k=round(self.percent * self.chromosome_length))
            ) / round(self.percent * self.chromosome_length)
