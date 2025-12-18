"""ATNF-Chat Benchmark Suite.

This package provides tools for evaluating LLM accuracy on pulsar queries.

Example:
    >>> from benchmarks import BenchmarkRunner, run_benchmarks
    >>> results = run_benchmarks()
    >>> print(results.summary())
"""

from benchmarks.evaluate import (
    BenchmarkResults,
    BenchmarkRunner,
    MockLLMResponse,
    ResponseEvaluator,
    TestCaseResult,
    run_benchmarks,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResults",
    "TestCaseResult",
    "ResponseEvaluator",
    "MockLLMResponse",
    "run_benchmarks",
]
