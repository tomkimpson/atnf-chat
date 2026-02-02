"""Benchmark evaluation framework for ATNF-Chat.

This module provides tools to evaluate LLM accuracy on pulsar queries,
including:
- Tool call accuracy
- DSL format compliance
- Parameter mapping accuracy
- Provenance completeness
- Failure case handling

Example:
    >>> from benchmarks.evaluate import BenchmarkRunner
    >>> runner = BenchmarkRunner()
    >>> results = runner.run_all()
    >>> print(results.summary())
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Path to benchmark test cases
BENCHMARK_FILE = Path(__file__).parent / "pulsar_queries.json"

# Semantic keywords used in expected_dsl patterns (medium/easy test cases)
SEMANTIC_KEYWORDS = {
    "must_have_field", "must_have_cmp", "value_range", "value_approximately",
    "value_contains", "clauses_must_include", "select_fields_must_include",
    "limit_range",
}


@dataclass
class TestCaseResult:
    """Result from evaluating a single test case."""

    test_id: str
    query: str
    difficulty: str
    test_type: str  # "standard", "failure", or "dsl"
    passed: bool
    scores: dict[str, float]
    details: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate overall score as average of individual scores."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)


@dataclass
class BenchmarkResults:
    """Aggregated results from a benchmark run."""

    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: list[TestCaseResult]
    scores_by_difficulty: dict[str, float]
    scores_by_type: dict[str, float]
    overall_score: float

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "ATNF-Chat Benchmark Results",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Total Tests: {self.total_tests}",
            f"Passed: {self.passed_tests} ({self.passed_tests/self.total_tests*100:.1f}%)",
            f"Failed: {self.failed_tests}",
            "",
            "Scores by Difficulty:",
        ]

        for difficulty, score in self.scores_by_difficulty.items():
            target = {"easy": 0.95, "medium": 0.85, "hard": 0.70}.get(difficulty, 0.8)
            status = "✓" if score >= target else "✗"
            lines.append(f"  {difficulty}: {score:.1%} (target: {target:.0%}) {status}")

        lines.extend([
            "",
            "Scores by Test Type:",
        ])

        for test_type, score in self.scores_by_type.items():
            lines.append(f"  {test_type}: {score:.1%}")

        lines.extend([
            "",
            f"Overall Score: {self.overall_score:.1%}",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "scores_by_difficulty": self.scores_by_difficulty,
            "scores_by_type": self.scores_by_type,
            "overall_score": self.overall_score,
            "test_results": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "difficulty": r.difficulty,
                    "test_type": r.test_type,
                    "passed": r.passed,
                    "scores": r.scores,
                    "overall_score": r.overall_score,
                    "errors": r.errors,
                    "warnings": r.warnings,
                }
                for r in self.test_results
            ],
        }


class ResponseEvaluator:
    """Evaluates LLM responses against expected outcomes."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self._load_benchmark_data()

    def _load_benchmark_data(self) -> None:
        """Load benchmark test cases from JSON file."""
        with open(BENCHMARK_FILE) as f:
            self.benchmark_data = json.load(f)

    def evaluate_tool_calls(
        self,
        expected_tools: list[str] | None,
        actual_tools: list[str],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate if correct tools were called.

        Args:
            expected_tools: List of expected tool names
            actual_tools: List of tools actually called

        Returns:
            Tuple of (score, details)
        """
        if expected_tools is None:
            return 1.0, {"note": "No expected tools specified"}

        if not actual_tools:
            return 0.0, {"error": "No tools called", "expected": expected_tools}

        # Check if all expected tools were called
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)

        correct = expected_set & actual_set
        missing = expected_set - actual_set
        extra = actual_set - expected_set

        # Score: proportion of expected tools called, penalize extras slightly
        if not expected_set:
            score = 1.0
        else:
            score = len(correct) / len(expected_set)
            if extra:
                score *= 0.9  # Small penalty for extra tools

        return score, {
            "expected": list(expected_set),
            "actual": list(actual_set),
            "correct": list(correct),
            "missing": list(missing),
            "extra": list(extra),
        }

    def evaluate_dsl_compliance(
        self,
        query_dsl: dict[str, Any] | None,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate if query uses valid DSL format.

        Args:
            query_dsl: The DSL query dict (if any)

        Returns:
            Tuple of (score, details)
        """
        if query_dsl is None:
            return 1.0, {"note": "No query DSL generated"}

        try:
            from atnf_chat.core.dsl import QueryDSL

            # Try to validate the DSL
            QueryDSL(**query_dsl)
            return 1.0, {"valid": True}

        except ValidationError as e:
            return 0.0, {
                "valid": False,
                "validation_errors": str(e),
            }
        except Exception as e:
            return 0.0, {
                "valid": False,
                "error": str(e),
            }

    def evaluate_dsl_pattern(
        self,
        query_dsl: dict[str, Any] | None,
        expected_pattern: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate if DSL matches expected pattern.

        Args:
            query_dsl: The generated DSL
            expected_pattern: Pattern to match against

        Returns:
            Tuple of (score, details)
        """
        if query_dsl is None:
            return 0.0, {"error": "No DSL generated"}

        if self._is_semantic_pattern(expected_pattern):
            return self._evaluate_semantic_dsl_pattern(query_dsl, expected_pattern)

        matches = []
        mismatches = []

        for path, expected_value in expected_pattern.items():
            actual_value = self._get_nested_value(query_dsl, path)

            if path.endswith("_count"):
                # Special handling for count checks
                base_path = path.rsplit("_", 1)[0]
                actual_list = self._get_nested_value(query_dsl, base_path)
                if isinstance(actual_list, list):
                    if len(actual_list) == expected_value:
                        matches.append(path)
                    else:
                        mismatches.append({
                            "path": path,
                            "expected": expected_value,
                            "actual": len(actual_list),
                        })
                else:
                    mismatches.append({
                        "path": path,
                        "error": "Not a list",
                    })
            elif actual_value == expected_value:
                matches.append(path)
            elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Allow small numeric tolerance
                if abs(actual_value - expected_value) / max(abs(expected_value), 1) < 0.1:
                    matches.append(path)
                else:
                    mismatches.append({
                        "path": path,
                        "expected": expected_value,
                        "actual": actual_value,
                    })
            else:
                mismatches.append({
                    "path": path,
                    "expected": expected_value,
                    "actual": actual_value,
                })

        total = len(matches) + len(mismatches)
        score = len(matches) / total if total > 0 else 1.0

        return score, {
            "matches": matches,
            "mismatches": mismatches,
        }

    def _get_nested_value(self, data: dict[str, Any], path: str) -> Any:
        """Get a nested value from a dict using dot notation with array indices.

        Args:
            data: Dictionary to traverse
            path: Dot-separated path with optional array indices (e.g., "filters.clauses[0].field")

        Returns:
            Value at path or None if not found
        """
        current = data
        parts = path.replace("[", ".").replace("]", "").split(".")

        for part in parts:
            if current is None:
                return None
            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def _is_semantic_pattern(self, pattern: dict[str, Any]) -> bool:
        """Check if a pattern uses semantic keywords rather than literal paths."""
        for key in pattern:
            if key in SEMANTIC_KEYWORDS:
                return True
        # Also check inside a nested "filters" sub-dict
        filters = pattern.get("filters")
        if isinstance(filters, dict):
            for key in filters:
                if key in SEMANTIC_KEYWORDS:
                    return True
        return False

    def _extract_all_clauses(self, dsl: dict[str, Any]) -> list[dict[str, Any]]:
        """Recursively flatten all FilterClauses from filters.clauses."""
        filters = dsl.get("filters")
        if not filters:
            return []
        return self._flatten_clauses(filters)

    def _flatten_clauses(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        """Recursively collect leaf clauses from a FilterGroup dict."""
        result = []
        for clause in group.get("clauses", []):
            if "clauses" in clause:
                # Nested FilterGroup
                result.extend(self._flatten_clauses(clause))
            else:
                result.append(clause)
        return result

    def _evaluate_semantic_dsl_pattern(
        self,
        query_dsl: dict[str, Any],
        expected_pattern: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate a DSL against a semantic expected_dsl pattern."""
        matches: list[str] = []
        mismatches: list[dict[str, Any]] = []
        all_clauses = self._extract_all_clauses(query_dsl)

        # --- Top-level checks ---

        # select_fields_must_include
        if "select_fields_must_include" in expected_pattern:
            required = expected_pattern["select_fields_must_include"]
            actual = query_dsl.get("select_fields")
            if actual is None:
                # None means "all fields" — that includes everything
                matches.append("select_fields_must_include")
            elif all(f in actual for f in required):
                matches.append("select_fields_must_include")
            else:
                mismatches.append({
                    "check": "select_fields_must_include",
                    "expected": required,
                    "actual": actual,
                })

        # limit_range
        if "limit_range" in expected_pattern:
            lo, hi = expected_pattern["limit_range"]
            actual_limit = query_dsl.get("limit")
            if actual_limit is not None and lo <= actual_limit <= hi:
                matches.append("limit_range")
            else:
                mismatches.append({
                    "check": "limit_range",
                    "expected": [lo, hi],
                    "actual": actual_limit,
                })

        # --- Filters sub-dict checks ---
        filters_pattern = expected_pattern.get("filters")
        if isinstance(filters_pattern, dict):
            self._check_filters_pattern(
                filters_pattern, query_dsl, all_clauses, matches, mismatches,
            )

        total = len(matches) + len(mismatches)
        score = len(matches) / total if total > 0 else 1.0
        return score, {"matches": matches, "mismatches": mismatches}

    def _check_filters_pattern(
        self,
        filters_pattern: dict[str, Any],
        query_dsl: dict[str, Any],
        all_clauses: list[dict[str, Any]],
        matches: list[str],
        mismatches: list[dict[str, Any]],
    ) -> None:
        """Evaluate semantic keywords inside a filters sub-pattern."""

        # op check (literal)
        if "op" in filters_pattern:
            actual_op = self._get_nested_value(query_dsl, "filters.op")
            # Normalise enum values
            if hasattr(actual_op, "value"):
                actual_op = actual_op.value
            if actual_op == filters_pattern["op"]:
                matches.append("filters.op")
            else:
                mismatches.append({
                    "check": "filters.op",
                    "expected": filters_pattern["op"],
                    "actual": actual_op,
                })

        # must_have_field + must_have_cmp  (find a clause matching both)
        if "must_have_field" in filters_pattern:
            req_field = filters_pattern["must_have_field"]
            req_cmp = filters_pattern.get("must_have_cmp")
            matched_clause = self._find_matching_clause(
                all_clauses, req_field, req_cmp,
            )
            if matched_clause is not None:
                matches.append(f"must_have_field={req_field}")
                if req_cmp:
                    matches.append(f"must_have_cmp={req_cmp}")

                # value_range
                if "value_range" in filters_pattern:
                    lo, hi = filters_pattern["value_range"]
                    val = matched_clause.get("value")
                    if isinstance(val, (int, float)) and lo <= val <= hi:
                        matches.append("value_range")
                    else:
                        mismatches.append({
                            "check": "value_range",
                            "expected": [lo, hi],
                            "actual": val,
                        })

                # value_approximately (for in_range clauses)
                if "value_approximately" in filters_pattern:
                    lo, hi = filters_pattern["value_approximately"]
                    val = matched_clause.get("value")
                    if isinstance(val, list) and len(val) == 2:
                        # Range overlaps with expected range
                        if val[0] <= hi and val[1] >= lo:
                            matches.append("value_approximately")
                        else:
                            mismatches.append({
                                "check": "value_approximately",
                                "expected": [lo, hi],
                                "actual": val,
                            })
                    elif isinstance(val, (int, float)) and lo <= val <= hi:
                        matches.append("value_approximately")
                    else:
                        mismatches.append({
                            "check": "value_approximately",
                            "expected": [lo, hi],
                            "actual": val,
                        })

                # value (exact, with 10% numeric tolerance)
                if "value" in filters_pattern:
                    expected_val = filters_pattern["value"]
                    actual_val = matched_clause.get("value")
                    if self._values_close(actual_val, expected_val):
                        matches.append("value")
                    else:
                        mismatches.append({
                            "check": "value",
                            "expected": expected_val,
                            "actual": actual_val,
                        })

                # value_contains
                if "value_contains" in filters_pattern:
                    substrings = filters_pattern["value_contains"]
                    actual_val = str(matched_clause.get("value", ""))
                    if any(s.lower() in actual_val.lower() for s in substrings):
                        matches.append("value_contains")
                    else:
                        mismatches.append({
                            "check": "value_contains",
                            "expected_any_of": substrings,
                            "actual": actual_val,
                        })
            else:
                mismatches.append({
                    "check": "must_have_field",
                    "expected_field": req_field,
                    "expected_cmp": req_cmp,
                    "actual_clauses": all_clauses,
                })

        # clauses_must_include (order-independent list of clause patterns)
        if "clauses_must_include" in filters_pattern:
            for i, clause_pat in enumerate(filters_pattern["clauses_must_include"]):
                req_field = clause_pat.get("field")
                req_cmp = clause_pat.get("cmp")
                matched = self._find_matching_clause(
                    all_clauses, req_field, req_cmp,
                )
                label = f"clauses_must_include[{i}](field={req_field})"
                if matched is not None:
                    matches.append(label)

                    # Nested value check
                    if "value" in clause_pat:
                        if self._values_close(matched.get("value"), clause_pat["value"]):
                            matches.append(f"{label}.value")
                        else:
                            mismatches.append({
                                "check": f"{label}.value",
                                "expected": clause_pat["value"],
                                "actual": matched.get("value"),
                            })

                    # Nested value_contains check
                    if "value_contains" in clause_pat:
                        substrings = clause_pat["value_contains"]
                        actual_val = str(matched.get("value", ""))
                        if any(s.lower() in actual_val.lower() for s in substrings):
                            matches.append(f"{label}.value_contains")
                        else:
                            mismatches.append({
                                "check": f"{label}.value_contains",
                                "expected_any_of": substrings,
                                "actual": actual_val,
                            })
                else:
                    mismatches.append({
                        "check": label,
                        "expected_field": req_field,
                        "expected_cmp": req_cmp,
                    })

    def _find_matching_clause(
        self,
        clauses: list[dict[str, Any]],
        field: str | None,
        cmp: str | None,
    ) -> dict[str, Any] | None:
        """Find a clause matching both field and cmp (if given)."""
        for clause in clauses:
            clause_field = clause.get("field", "")
            clause_cmp = clause.get("cmp", "")
            # Normalise enum values
            if hasattr(clause_cmp, "value"):
                clause_cmp = clause_cmp.value
            if field and clause_field.upper() != field.upper():
                continue
            if cmp and clause_cmp != cmp:
                continue
            return clause
        return None

    @staticmethod
    def _values_close(actual: Any, expected: Any) -> bool:
        """Compare values with 10% numeric tolerance."""
        if actual == expected:
            return True
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected == 0:
                return abs(actual) < 1e-9
            return abs(actual - expected) / max(abs(expected), 1) < 0.1
        return False

    def evaluate_provenance(
        self,
        provenance: dict[str, Any] | None,
        expected: dict[str, Any] | None,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate provenance completeness.

        Args:
            provenance: Actual provenance dict
            expected: Expected provenance requirements

        Returns:
            Tuple of (score, details)
        """
        required_fields = [
            "catalogue_version",
            "snapshot_date",
            "result_count",
            "null_counts",
        ]

        if provenance is None:
            return 0.0, {"error": "No provenance provided"}

        present = []
        missing = []

        for field in required_fields:
            if field in provenance and provenance[field] is not None:
                present.append(field)
            else:
                missing.append(field)

        # Check expected source if specified
        if expected and "source" in expected:
            if provenance.get("source") == expected["source"]:
                present.append("source_match")
            else:
                missing.append(f"source (expected: {expected['source']})")

        score = len(present) / (len(present) + len(missing)) if (present or missing) else 1.0

        return score, {
            "present": present,
            "missing": missing,
        }

    def evaluate_answer_content(
        self,
        answer: str,
        expected_contains: list[str] | None,
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate if answer contains expected content.

        Args:
            answer: The text answer
            expected_contains: List of strings that should appear

        Returns:
            Tuple of (score, details)
        """
        if expected_contains is None:
            return 1.0, {"note": "No expected content specified"}

        if not answer:
            return 0.0, {"error": "Empty answer"}

        answer_lower = answer.lower()
        found = []
        not_found = []

        for expected in expected_contains:
            if expected.lower() in answer_lower:
                found.append(expected)
            else:
                not_found.append(expected)

        score = len(found) / len(expected_contains) if expected_contains else 1.0

        return score, {
            "found": found,
            "not_found": not_found,
        }

    def evaluate_failure_handling(
        self,
        response: dict[str, Any],
        test_case: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate handling of failure cases.

        Args:
            response: The LLM response
            test_case: The test case definition

        Returns:
            Tuple of (score, details)
        """
        test_type = test_case.get("test_type", "unknown")
        expected_behavior = test_case.get("expected_behavior", "")
        answer = response.get("answer", "")
        answer_lower = answer.lower() if answer else ""

        checks_passed = []
        checks_failed = []

        if test_type == "ambiguous_term" or test_type == "vague_term":
            # Should ask for clarification
            clarification_indicators = [
                "would you like",
                "do you mean",
                "could you clarify",
                "which",
                "?",
            ]
            if any(ind in answer_lower for ind in clarification_indicators):
                checks_passed.append("asks_clarification")
            else:
                checks_failed.append("should_ask_clarification")

            # Check for clarification options
            if test_case.get("expected_clarification_options"):
                options_found = sum(
                    1 for opt in test_case["expected_clarification_options"]
                    if opt.lower() in answer_lower
                )
                if options_found >= 1:
                    checks_passed.append("offers_options")
                else:
                    checks_failed.append("should_offer_options")

        elif test_type == "empty_result":
            # Should gracefully handle empty results
            empty_indicators = ["no pulsars", "no results", "none found", "0 pulsars", "no matches"]
            if any(ind in answer_lower for ind in empty_indicators):
                checks_passed.append("handles_empty")
            else:
                checks_failed.append("should_acknowledge_empty")

            # Should suggest alternatives
            suggestion_indicators = ["try", "consider", "relax", "broaden", "alternative"]
            if any(ind in answer_lower for ind in suggestion_indicators):
                checks_passed.append("suggests_alternative")
            else:
                checks_failed.append("should_suggest_alternative")

        elif test_type == "missingness_handling" or test_type == "high_missingness":
            # Should warn about missing data
            warning_indicators = ["missing", "limited", "sparse", "only", "few", "incomplete"]
            if any(ind in answer_lower for ind in warning_indicators):
                checks_passed.append("warns_missingness")
            else:
                checks_failed.append("should_warn_missingness")

        elif test_type == "overly_broad":
            # Should provide focused summary
            if len(answer) > 100:  # Has substantial content
                checks_passed.append("provides_summary")
            else:
                checks_failed.append("should_provide_summary")

            # Should offer to focus
            focus_indicators = ["would you like", "want me to", "can focus", "specific"]
            if any(ind in answer_lower for ind in focus_indicators):
                checks_passed.append("offers_focus")
            else:
                checks_failed.append("should_offer_focus")

        elif test_type == "not_found":
            # Should handle not found gracefully
            not_found_indicators = ["not found", "no match", "couldn't find", "doesn't exist"]
            if any(ind in answer_lower for ind in not_found_indicators):
                checks_passed.append("handles_not_found")
            else:
                checks_failed.append("should_acknowledge_not_found")

        elif test_type == "unsupported_query":
            # Should explain limitation
            limitation_indicators = [
                "not available",
                "not tracked",
                "cannot",
                "doesn't include",
                "limitation",
            ]
            if any(ind in answer_lower for ind in limitation_indicators):
                checks_passed.append("explains_limitation")
            else:
                checks_failed.append("should_explain_limitation")

        total = len(checks_passed) + len(checks_failed)
        score = len(checks_passed) / total if total > 0 else 0.5

        return score, {
            "test_type": test_type,
            "expected_behavior": expected_behavior,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
        }


class MockLLMResponse:
    """Mock LLM response for testing the benchmark framework."""

    def __init__(
        self,
        answer: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        query_dsl: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.answer = answer
        self.tool_calls = tool_calls or []
        self.query_dsl = query_dsl
        self.provenance = provenance

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "tool_calls": self.tool_calls,
            "query_dsl": self.query_dsl,
            "provenance": self.provenance,
        }


class BenchmarkRunner:
    """Runs the full benchmark suite."""

    def __init__(
        self,
        llm_client: Any = None,
        benchmark_file: Path | None = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            llm_client: LLM client for generating responses (None for dry run)
            benchmark_file: Path to benchmark JSON (default: bundled file)
        """
        self.llm_client = llm_client
        self.benchmark_file = benchmark_file or BENCHMARK_FILE
        self.evaluator = ResponseEvaluator()

        with open(self.benchmark_file) as f:
            self.benchmark_data = json.load(f)

    def run_all(self, skip_llm: bool = False) -> BenchmarkResults:
        """Run all benchmark tests.

        Args:
            skip_llm: If True, skip actual LLM calls (for testing framework)

        Returns:
            BenchmarkResults with all test outcomes
        """
        results: list[TestCaseResult] = []

        # Run standard test cases
        for test_case in self.benchmark_data.get("test_cases", []):
            result = self._run_test_case(test_case, "standard", skip_llm)
            results.append(result)

        # Run failure case tests
        for test_case in self.benchmark_data.get("failure_case_tests", []):
            result = self._run_test_case(test_case, "failure", skip_llm)
            results.append(result)

        # Run DSL compliance tests
        for test_case in self.benchmark_data.get("dsl_compliance_tests", []):
            result = self._run_dsl_test(test_case, skip_llm)
            results.append(result)

        return self._aggregate_results(results)

    def _run_test_case(
        self,
        test_case: dict[str, Any],
        test_type: str,
        skip_llm: bool,
    ) -> TestCaseResult:
        """Run a single test case.

        Args:
            test_case: Test case definition
            test_type: "standard" or "failure"
            skip_llm: Skip LLM call

        Returns:
            TestCaseResult
        """
        test_id = test_case.get("id", "unknown")
        query = test_case.get("query", "")
        difficulty = test_case.get("difficulty", "medium")

        scores: dict[str, float] = {}
        details: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Get LLM response (or mock for testing)
            if skip_llm or self.llm_client is None:
                response = self._create_mock_response(test_case)
            else:
                response = self._get_llm_response(query)

            response_dict = response.to_dict() if hasattr(response, "to_dict") else response

            # Evaluate tool calls
            score, detail = self.evaluator.evaluate_tool_calls(
                test_case.get("expected_tool_calls"),
                [tc.get("name", "") for tc in response_dict.get("tool_calls", [])],
            )
            scores["tool_calls"] = score
            details["tool_calls"] = detail

            # Evaluate DSL compliance
            query_dsl = response_dict.get("query_dsl")
            score, detail = self.evaluator.evaluate_dsl_compliance(query_dsl)
            scores["dsl_compliance"] = score
            details["dsl_compliance"] = detail

            # Evaluate DSL pattern if expected
            if "expected_dsl" in test_case and query_dsl:
                score, detail = self.evaluator.evaluate_dsl_pattern(
                    query_dsl,
                    test_case["expected_dsl"],
                )
                scores["dsl_pattern"] = score
                details["dsl_pattern"] = detail

            # Evaluate provenance
            score, detail = self.evaluator.evaluate_provenance(
                response_dict.get("provenance"),
                test_case.get("expected_provenance"),
            )
            scores["provenance"] = score
            details["provenance"] = detail

            # Evaluate answer content
            score, detail = self.evaluator.evaluate_answer_content(
                response_dict.get("answer", ""),
                test_case.get("expected_answer_contains"),
            )
            scores["answer_content"] = score
            details["answer_content"] = detail

            # For failure cases, evaluate failure handling
            if test_type == "failure":
                score, detail = self.evaluator.evaluate_failure_handling(
                    response_dict,
                    test_case,
                )
                scores["failure_handling"] = score
                details["failure_handling"] = detail

        except Exception as e:
            logger.exception(f"Error running test {test_id}")
            errors.append(str(e))
            scores["error"] = 0.0

        # Determine if test passed
        # Standard tests: average score >= 0.7
        # Failure tests: failure_handling score >= 0.7
        if test_type == "failure":
            passed = scores.get("failure_handling", 0) >= 0.7
        else:
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            passed = avg_score >= 0.7

        return TestCaseResult(
            test_id=test_id,
            query=query,
            difficulty=difficulty,
            test_type=test_type,
            passed=passed,
            scores=scores,
            details=details,
            errors=errors,
            warnings=warnings,
        )

    def _run_dsl_test(
        self,
        test_case: dict[str, Any],
        skip_llm: bool,
    ) -> TestCaseResult:
        """Run a DSL compliance test.

        Args:
            test_case: Test case definition
            skip_llm: Skip LLM call

        Returns:
            TestCaseResult
        """
        test_id = test_case.get("id", "unknown")
        query = test_case.get("input_query", "")

        scores: dict[str, float] = {}
        details: dict[str, Any] = {"description": test_case.get("description", "")}
        errors: list[str] = []

        try:
            if skip_llm or self.llm_client is None:
                response = self._create_mock_dsl_response(test_case)
            else:
                response = self._get_llm_response(query)

            response_dict = response.to_dict() if hasattr(response, "to_dict") else response
            query_dsl = response_dict.get("query_dsl")

            # Evaluate DSL compliance
            score, detail = self.evaluator.evaluate_dsl_compliance(query_dsl)
            scores["dsl_compliance"] = score
            details["dsl_compliance"] = detail

            # Evaluate DSL pattern
            if "expected_dsl_pattern" in test_case:
                score, detail = self.evaluator.evaluate_dsl_pattern(
                    query_dsl,
                    test_case["expected_dsl_pattern"],
                )
                scores["dsl_pattern"] = score
                details["dsl_pattern"] = detail

        except Exception as e:
            logger.exception(f"Error running DSL test {test_id}")
            errors.append(str(e))
            scores["error"] = 0.0

        avg_score = sum(scores.values()) / len(scores) if scores else 0
        passed = avg_score >= 0.8  # Higher threshold for DSL compliance

        return TestCaseResult(
            test_id=test_id,
            query=query,
            difficulty="dsl",
            test_type="dsl",
            passed=passed,
            scores=scores,
            details=details,
            errors=errors,
        )

    def _get_llm_response(self, query: str) -> dict[str, Any]:
        """Get response from LLM client.

        Args:
            query: User query

        Returns:
            Response dictionary
        """
        # This would integrate with the actual chat endpoint
        # For now, raise NotImplementedError
        raise NotImplementedError("LLM client integration not implemented")

    def _create_mock_response(self, test_case: dict[str, Any]) -> MockLLMResponse:
        """Create a mock response based on test case expectations.

        This is used for testing the benchmark framework itself.
        """
        # Create a "perfect" response based on expectations
        tool_calls = [
            {"name": tool} for tool in test_case.get("expected_tool_calls", [])
        ]

        # Build mock DSL from expected_dsl hints
        query_dsl = None
        if "expected_dsl" in test_case:
            query_dsl = self._build_mock_dsl_from_semantic(test_case["expected_dsl"])

        # Build mock provenance
        provenance = {
            "catalogue_version": "2.0.0",
            "snapshot_date": datetime.now().isoformat(),
            "result_count": 100,
            "null_counts": {},
            "source": test_case.get("expected_provenance", {}).get("source", "atnf_native"),
        }

        # Build mock answer
        answer_parts = test_case.get("expected_answer_contains", [])
        answer = " ".join(answer_parts) if answer_parts else "Mock response"

        return MockLLMResponse(
            answer=answer,
            tool_calls=tool_calls,
            query_dsl=query_dsl,
            provenance=provenance,
        )

    def _build_mock_dsl_from_semantic(self, expected: dict[str, Any]) -> dict[str, Any]:
        """Build a valid DSL dict from semantic expected_dsl patterns."""
        dsl: dict[str, Any] = {}

        # select_fields
        if "select_fields_must_include" in expected:
            dsl["select_fields"] = expected["select_fields_must_include"]
        else:
            dsl["select_fields"] = None

        # limit
        if "limit_range" in expected:
            dsl["limit"] = expected["limit_range"][0]
        else:
            dsl["limit"] = None

        # Top-level literal keys (e.g. order_by, order_desc for hard cases)
        for key in ("order_by", "order_desc"):
            if key in expected:
                dsl[key] = expected[key]

        # Build filters from semantic patterns
        clauses: list[dict[str, Any]] = []
        filters_pat = expected.get("filters")
        if isinstance(filters_pat, dict):
            op = filters_pat.get("op", "and")

            # must_have_field / must_have_cmp
            if "must_have_field" in filters_pat:
                clause: dict[str, Any] = {
                    "field": filters_pat["must_have_field"],
                    "cmp": filters_pat.get("must_have_cmp", "lt"),
                }
                # Pick a value that satisfies the pattern checks
                if "value_range" in filters_pat:
                    lo, hi = filters_pat["value_range"]
                    clause["value"] = (lo + hi) / 2
                elif "value_approximately" in filters_pat:
                    lo, hi = filters_pat["value_approximately"]
                    clause["value"] = [lo, hi]
                elif "value" in filters_pat:
                    clause["value"] = filters_pat["value"]
                elif "value_contains" in filters_pat:
                    clause["value"] = filters_pat["value_contains"][0]
                clauses.append(clause)

            # clauses_must_include
            if "clauses_must_include" in filters_pat:
                for pat in filters_pat["clauses_must_include"]:
                    c: dict[str, Any] = {
                        "field": pat["field"],
                        "cmp": pat.get("cmp", "eq"),
                    }
                    if "value" in pat:
                        c["value"] = pat["value"]
                    elif "value_contains" in pat:
                        c["value"] = pat["value_contains"][0]
                    clauses.append(c)

            if clauses:
                dsl["filters"] = {"op": op, "clauses": clauses}
            else:
                dsl["filters"] = None
        else:
            dsl["filters"] = None

        return dsl

    def _create_mock_dsl_response(self, test_case: dict[str, Any]) -> MockLLMResponse:
        """Create mock DSL response for DSL compliance tests."""
        pattern = test_case.get("expected_dsl_pattern", {})

        # Build DSL from pattern
        query_dsl: dict[str, Any] = {"filters": {"op": "and", "clauses": []}}

        for key, value in pattern.items():
            if key.startswith("filters.clauses["):
                # Handle both "clauses[0].field" and "clauses[0].value[1]"
                match = re.match(
                    r"filters\.clauses\[(\d+)\]\.(\w+)(?:\[(\d+)\])?", key,
                )
                if match:
                    idx = int(match.group(1))
                    field_name = match.group(2)
                    arr_idx = match.group(3)

                    # Ensure clause exists
                    while len(query_dsl["filters"]["clauses"]) <= idx:
                        query_dsl["filters"]["clauses"].append({})

                    if arr_idx is not None:
                        # Array element (e.g. value[0], value[1])
                        arr_idx = int(arr_idx)
                        existing = query_dsl["filters"]["clauses"][idx].get(field_name)
                        if not isinstance(existing, list):
                            existing = []
                        while len(existing) <= arr_idx:
                            existing.append(None)
                        existing[arr_idx] = value
                        query_dsl["filters"]["clauses"][idx][field_name] = existing
                    else:
                        query_dsl["filters"]["clauses"][idx][field_name] = value
            elif key == "filters.clauses_count":
                # Generate placeholder clauses to satisfy the count
                target = value
                clauses = query_dsl["filters"]["clauses"]
                while len(clauses) < target:
                    clauses.append({"field": "P0", "cmp": "lt", "value": 0.03})
            elif key == "filters.op":
                query_dsl["filters"]["op"] = value

        # Ensure operators that require a value have one
        for clause in query_dsl["filters"]["clauses"]:
            cmp = clause.get("cmp", "")
            if cmp in ("contains", "startswith") and "value" not in clause:
                clause["value"] = "placeholder"
            elif cmp in ("eq", "ne", "lt", "le", "gt", "ge") and "value" not in clause:
                clause["value"] = 0

        return MockLLMResponse(
            answer="Mock DSL response",
            tool_calls=[{"name": "query_catalogue"}],
            query_dsl=query_dsl,
            provenance={
                "catalogue_version": "2.0.0",
                "snapshot_date": datetime.now().isoformat(),
                "result_count": 100,
                "null_counts": {},
            },
        )

    def _aggregate_results(self, results: list[TestCaseResult]) -> BenchmarkResults:
        """Aggregate individual test results.

        Args:
            results: List of test case results

        Returns:
            BenchmarkResults summary
        """
        # Count passed/failed
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        # Score by difficulty
        by_difficulty: dict[str, list[float]] = {}
        for r in results:
            if r.difficulty not in by_difficulty:
                by_difficulty[r.difficulty] = []
            by_difficulty[r.difficulty].append(r.overall_score)

        scores_by_difficulty = {
            diff: sum(scores) / len(scores) if scores else 0.0
            for diff, scores in by_difficulty.items()
        }

        # Score by type
        by_type: dict[str, list[float]] = {}
        for r in results:
            if r.test_type not in by_type:
                by_type[r.test_type] = []
            by_type[r.test_type].append(r.overall_score)

        scores_by_type = {
            t: sum(scores) / len(scores) if scores else 0.0
            for t, scores in by_type.items()
        }

        # Overall score
        overall = sum(r.overall_score for r in results) / len(results) if results else 0.0

        return BenchmarkResults(
            timestamp=datetime.now().isoformat(),
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            test_results=results,
            scores_by_difficulty=scores_by_difficulty,
            scores_by_type=scores_by_type,
            overall_score=overall,
        )


def run_benchmarks(output_file: Path | None = None) -> BenchmarkResults:
    """Run benchmarks and optionally save results.

    Args:
        output_file: Path to save JSON results (optional)

    Returns:
        BenchmarkResults
    """
    runner = BenchmarkRunner()
    results = runner.run_all(skip_llm=True)  # Skip LLM for now

    print(results.summary())

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    import sys

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_benchmarks(output)
