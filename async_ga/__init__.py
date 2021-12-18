"""
A python package for asynchronous genetic algorithms.

Motivation:
    If a chromosome's fitness is repeatedly estimateable, but
    its exact value cannot be computed, or if a chromosome is
    changing over time, then AsyncGA will be able to concurrently
    update the estimate of the chromosome's fitness while evolving.

### Usage
For usage, see `help(async_ga.AsyncGA)`.

### Example
For the example documentation, see `help(async_ga.DefaultGA)`.
    >>> from async_ga import DefaultGA
    >>> import asyncio
    >>> async def get_best_chromosome():
    ...     ga = DefaultGA()
    ...     async for population in ga.evolve():
    ...         print(population[0].fitness.mean)
    ...     return population[0]
    ...
    >>> chromosome = asyncio.run(get_best_chromosome())
      ...
    >>> print(chromosome)
      ...
"""
from async_ga.chromosome import *
from async_ga.crosser import *
from async_ga.filter import *
from async_ga.fitness.function import *
from async_ga.fitness.maturity import *
from async_ga.ga import *
from async_ga.mutator import *
from async_ga.selector import *
from async_ga.terminator import *

from types import FunctionType
from typing import get_type_hints

# Finalize type-hints.
for obj in list(vars().values()):
    if obj is FunctionType or obj is get_type_hints:
        continue
    elif isinstance(obj, FunctionType):
        obj.__annotations__ = get_type_hints(obj)
    elif not isinstance(obj, type):
        continue
    obj.__annotations__ = get_type_hints(obj)
    for name, method in vars(obj).items():
        if isinstance(method, classmethod):
            method.__func__.__annotations__ = get_type_hints(method.__func__)
        elif isinstance(method, FunctionType):
            method.__annotations__ = get_type_hints(method)

del FunctionType, get_type_hints, method, name, obj
