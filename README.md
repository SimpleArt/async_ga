# AsyncGA
 
A python package for asynchronous genetic algorithms.

### Motivation
If a chromosome's fitness is repeatedly estimateable, but its exact value cannot be computed, or if a chromosome is changing over time, then AsyncGA will be able to concurrently update the estimate of the chromosome's fitness while evolving.

### Usage
For usage, see `help(async_ga.AsyncGA)`.
For an example, see `help(async_ga.DefaultGA)`.
