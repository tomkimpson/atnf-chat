"""Tests for the benchmark evaluation framework."""

import json
from pathlib import Path

import pytest

from benchmarks.evaluate import (
    BENCHMARK_FILE,
    BenchmarkRunner,
    MockLLMResponse,
    ResponseEvaluator,
)


class TestBenchmarkFileStructure:
    """Test that the benchmark file is valid and complete."""

    @pytest.fixture
    def benchmark_data(self) -> dict:
        """Load the benchmark data."""
        with open(BENCHMARK_FILE) as f:
            return json.load(f)

    def test_benchmark_file_exists(self) -> None:
        """Benchmark file should exist."""
        assert BENCHMARK_FILE.exists()

    def test_has_required_sections(self, benchmark_data: dict) -> None:
        """Benchmark file should have all required sections."""
        assert "test_cases" in benchmark_data
        assert "failure_case_tests" in benchmark_data
        assert "dsl_compliance_tests" in benchmark_data
        assert "target_performance" in benchmark_data

    def test_test_cases_have_required_fields(self, benchmark_data: dict) -> None:
        """All test cases should have required fields."""
        required_fields = ["id", "query", "difficulty"]

        for test in benchmark_data["test_cases"]:
            for field in required_fields:
                assert field in test, f"Test {test.get('id', '?')} missing {field}"

    def test_failure_cases_have_required_fields(self, benchmark_data: dict) -> None:
        """All failure cases should have required fields."""
        required_fields = ["id", "query", "test_type", "expected_behavior"]

        for test in benchmark_data["failure_case_tests"]:
            for field in required_fields:
                assert field in test, f"Test {test.get('id', '?')} missing {field}"

    def test_difficulty_levels_valid(self, benchmark_data: dict) -> None:
        """All difficulty levels should be valid."""
        valid_difficulties = {"easy", "medium", "hard"}

        for test in benchmark_data["test_cases"]:
            assert test["difficulty"] in valid_difficulties, (
                f"Test {test['id']} has invalid difficulty: {test['difficulty']}"
            )

    def test_has_tests_for_each_difficulty(self, benchmark_data: dict) -> None:
        """Should have tests for each difficulty level."""
        difficulties = {test["difficulty"] for test in benchmark_data["test_cases"]}

        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_unique_test_ids(self, benchmark_data: dict) -> None:
        """All test IDs should be unique."""
        all_ids = []

        for test in benchmark_data["test_cases"]:
            all_ids.append(test["id"])

        for test in benchmark_data["failure_case_tests"]:
            all_ids.append(test["id"])

        for test in benchmark_data["dsl_compliance_tests"]:
            all_ids.append(test["id"])

        assert len(all_ids) == len(set(all_ids)), "Duplicate test IDs found"


class TestResponseEvaluator:
    """Test the response evaluator."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        """Create an evaluator instance."""
        return ResponseEvaluator()

    def test_evaluate_tool_calls_exact_match(self, evaluator: ResponseEvaluator) -> None:
        """Perfect tool call match should score 1.0."""
        expected = ["query_catalogue", "statistical_analysis"]
        actual = ["query_catalogue", "statistical_analysis"]

        score, details = evaluator.evaluate_tool_calls(expected, actual)

        assert score == 1.0
        assert not details.get("missing")
        assert not details.get("extra")

    def test_evaluate_tool_calls_missing(self, evaluator: ResponseEvaluator) -> None:
        """Missing tool calls should reduce score."""
        expected = ["query_catalogue", "statistical_analysis"]
        actual = ["query_catalogue"]

        score, details = evaluator.evaluate_tool_calls(expected, actual)

        assert score == 0.5
        assert "statistical_analysis" in details["missing"]

    def test_evaluate_tool_calls_extra(self, evaluator: ResponseEvaluator) -> None:
        """Extra tool calls should slightly reduce score."""
        expected = ["query_catalogue"]
        actual = ["query_catalogue", "extra_tool"]

        score, details = evaluator.evaluate_tool_calls(expected, actual)

        assert score == 0.9  # Small penalty for extra tools
        assert "extra_tool" in details["extra"]

    def test_evaluate_tool_calls_empty(self, evaluator: ResponseEvaluator) -> None:
        """No tools called when expected should score 0."""
        expected = ["query_catalogue"]
        actual: list[str] = []

        score, details = evaluator.evaluate_tool_calls(expected, actual)

        assert score == 0.0
        assert "No tools called" in details["error"]

    def test_evaluate_dsl_compliance_valid(self, evaluator: ResponseEvaluator) -> None:
        """Valid DSL should score 1.0."""
        query_dsl = {
            "select_fields": ["JNAME", "P0"],
            "filters": {
                "op": "and",
                "clauses": [{"field": "P0", "cmp": "lt", "value": 0.03}],
            },
        }

        score, details = evaluator.evaluate_dsl_compliance(query_dsl)

        assert score == 1.0
        assert details["valid"]

    def test_evaluate_dsl_compliance_invalid(self, evaluator: ResponseEvaluator) -> None:
        """Invalid DSL should score 0."""
        query_dsl = {
            "filters": {
                "op": "invalid_op",  # Invalid operator
                "clauses": [],
            },
        }

        score, details = evaluator.evaluate_dsl_compliance(query_dsl)

        assert score == 0.0
        assert not details["valid"]

    def test_evaluate_dsl_compliance_none(self, evaluator: ResponseEvaluator) -> None:
        """None DSL should score 1.0 (no query needed)."""
        score, details = evaluator.evaluate_dsl_compliance(None)

        assert score == 1.0
        assert "No query DSL" in details["note"]

    def test_evaluate_provenance_complete(self, evaluator: ResponseEvaluator) -> None:
        """Complete provenance should score 1.0."""
        provenance = {
            "catalogue_version": "2.0.0",
            "snapshot_date": "2024-01-01",
            "result_count": 100,
            "null_counts": {"P0": 5},
        }

        score, details = evaluator.evaluate_provenance(provenance, None)

        assert score == 1.0
        assert "catalogue_version" in details["present"]
        assert not details["missing"]

    def test_evaluate_provenance_missing_fields(self, evaluator: ResponseEvaluator) -> None:
        """Missing provenance fields should reduce score."""
        provenance = {
            "catalogue_version": "2.0.0",
            # Missing other fields
        }

        score, details = evaluator.evaluate_provenance(provenance, None)

        assert score < 1.0
        assert "snapshot_date" in details["missing"]

    def test_evaluate_provenance_none(self, evaluator: ResponseEvaluator) -> None:
        """None provenance should score 0."""
        score, details = evaluator.evaluate_provenance(None, None)

        assert score == 0.0
        assert "No provenance" in details["error"]

    def test_evaluate_answer_content_contains_all(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """Answer containing all expected strings should score 1.0."""
        answer = "The period is 0.089 seconds (89 milliseconds)."
        expected = ["0.089", "seconds"]

        score, details = evaluator.evaluate_answer_content(answer, expected)

        assert score == 1.0
        assert len(details["found"]) == 2
        assert not details["not_found"]

    def test_evaluate_answer_content_partial(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """Partial match should give partial score."""
        answer = "The period is 89 ms."
        expected = ["0.089", "89", "period"]

        score, details = evaluator.evaluate_answer_content(answer, expected)

        assert 0 < score < 1.0
        assert "89" in details["found"]
        assert "period" in details["found"]
        assert "0.089" in details["not_found"]

    def test_evaluate_answer_content_case_insensitive(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """String matching should be case-insensitive."""
        answer = "The PERIOD is measured."
        expected = ["period"]

        score, details = evaluator.evaluate_answer_content(answer, expected)

        assert score == 1.0

    def test_get_nested_value_simple(self, evaluator: ResponseEvaluator) -> None:
        """Simple path should work."""
        data = {"field": "value"}

        result = evaluator._get_nested_value(data, "field")

        assert result == "value"

    def test_get_nested_value_nested(self, evaluator: ResponseEvaluator) -> None:
        """Nested path should work."""
        data = {"outer": {"inner": "value"}}

        result = evaluator._get_nested_value(data, "outer.inner")

        assert result == "value"

    def test_get_nested_value_array(self, evaluator: ResponseEvaluator) -> None:
        """Array index path should work."""
        data = {"items": [{"name": "first"}, {"name": "second"}]}

        result = evaluator._get_nested_value(data, "items[1].name")

        assert result == "second"


class TestFailureHandling:
    """Test failure case handling evaluation."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        return ResponseEvaluator()

    def test_ambiguous_term_asks_clarification(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """Ambiguous term should trigger clarification check."""
        response = {
            "answer": "Would you like characteristic age or true age? I can calculate either."
        }
        test_case = {
            "test_type": "ambiguous_term",
            "expected_behavior": "clarifying_question",
        }

        score, details = evaluator.evaluate_failure_handling(response, test_case)

        assert "asks_clarification" in details["checks_passed"]

    def test_empty_result_handles_gracefully(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """Empty result should be handled gracefully."""
        response = {
            "answer": "No pulsars found matching those criteria. Consider relaxing the orbital period constraint."
        }
        test_case = {
            "test_type": "empty_result",
            "expected_behavior": "graceful_failure",
        }

        score, details = evaluator.evaluate_failure_handling(response, test_case)

        assert "handles_empty" in details["checks_passed"]
        assert "suggests_alternative" in details["checks_passed"]

    def test_missingness_warning(self, evaluator: ResponseEvaluator) -> None:
        """High missingness should trigger warning."""
        response = {
            "answer": "Only 12 pulsars have measured braking index, so results are sparse."
        }
        test_case = {
            "test_type": "missingness_handling",
            "expected_behavior": "execute_with_warning",
        }

        score, details = evaluator.evaluate_failure_handling(response, test_case)

        assert "warns_missingness" in details["checks_passed"]


class TestBenchmarkRunner:
    """Test the benchmark runner."""

    def test_runner_initialization(self) -> None:
        """Runner should initialize successfully."""
        runner = BenchmarkRunner()

        assert runner.benchmark_data is not None
        assert "test_cases" in runner.benchmark_data

    def test_run_all_dry_run(self) -> None:
        """Dry run should complete without errors."""
        runner = BenchmarkRunner()

        results = runner.run_all(skip_llm=True)

        assert results.total_tests > 0
        assert results.timestamp is not None

    def test_results_have_scores(self) -> None:
        """Results should include scores for each difficulty."""
        runner = BenchmarkRunner()

        results = runner.run_all(skip_llm=True)

        assert "easy" in results.scores_by_difficulty
        assert "medium" in results.scores_by_difficulty
        assert "hard" in results.scores_by_difficulty

    def test_results_summary(self) -> None:
        """Results should generate a summary."""
        runner = BenchmarkRunner()

        results = runner.run_all(skip_llm=True)
        summary = results.summary()

        assert "Benchmark Results" in summary
        assert "Scores by Difficulty" in summary

    def test_results_to_dict(self) -> None:
        """Results should serialize to dict."""
        runner = BenchmarkRunner()

        results = runner.run_all(skip_llm=True)
        data = results.to_dict()

        assert "timestamp" in data
        assert "total_tests" in data
        assert "test_results" in data


class TestMockLLMResponse:
    """Test the mock LLM response class."""

    def test_mock_response_to_dict(self) -> None:
        """Mock response should convert to dict."""
        response = MockLLMResponse(
            answer="Test answer",
            tool_calls=[{"name": "query_catalogue"}],
            query_dsl={"select_fields": ["JNAME"]},
            provenance={"catalogue_version": "2.0.0"},
        )

        data = response.to_dict()

        assert data["answer"] == "Test answer"
        assert len(data["tool_calls"]) == 1
        assert data["query_dsl"]["select_fields"] == ["JNAME"]

    def test_mock_response_defaults(self) -> None:
        """Mock response should have sensible defaults."""
        response = MockLLMResponse()

        data = response.to_dict()

        assert data["answer"] == ""
        assert data["tool_calls"] == []
        assert data["query_dsl"] is None


class TestSemanticDSLPattern:
    """Test the semantic DSL pattern matching evaluator."""

    @pytest.fixture
    def evaluator(self) -> ResponseEvaluator:
        return ResponseEvaluator()

    def test_must_have_field_and_cmp_match(self, evaluator: ResponseEvaluator) -> None:
        """Clause with matching field+cmp should score 1.0."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "P0", "cmp": "lt", "value": 0.02},
        ]}}
        pattern = {"filters": {"must_have_field": "P0", "must_have_cmp": "lt"}}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0
        assert any("must_have_field" in m for m in details["matches"])

    def test_must_have_field_no_match(self, evaluator: ResponseEvaluator) -> None:
        """Missing field should produce mismatches."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "DM", "cmp": "gt", "value": 10},
        ]}}
        pattern = {"filters": {"must_have_field": "P0", "must_have_cmp": "lt"}}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 0.0
        assert len(details["mismatches"]) > 0

    def test_value_range_in_range(self, evaluator: ResponseEvaluator) -> None:
        """Value within range should pass."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "P0", "cmp": "lt", "value": 0.02},
        ]}}
        pattern = {"filters": {
            "must_have_field": "P0", "must_have_cmp": "lt",
            "value_range": [0.01, 0.03],
        }}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0
        assert "value_range" in details["matches"]

    def test_value_range_out_of_range(self, evaluator: ResponseEvaluator) -> None:
        """Value outside range should fail that check."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "P0", "cmp": "lt", "value": 0.5},
        ]}}
        pattern = {"filters": {
            "must_have_field": "P0", "must_have_cmp": "lt",
            "value_range": [0.01, 0.03],
        }}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score < 1.0
        assert any(m.get("check") == "value_range" for m in details["mismatches"])

    def test_value_approximately_with_range_clause(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """in_range clause overlapping expected range should pass."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "P0", "cmp": "in_range", "value": [0.0013, 0.0015]},
        ]}}
        pattern = {"filters": {
            "must_have_field": "P0", "must_have_cmp": "in_range",
            "value_approximately": [0.0013, 0.0015],
        }}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0
        assert "value_approximately" in details["matches"]

    def test_value_contains(self, evaluator: ResponseEvaluator) -> None:
        """String value containing expected substring should pass."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "ASSOC", "cmp": "contains", "value": "SNR"},
        ]}}
        pattern = {"filters": {
            "must_have_field": "ASSOC", "must_have_cmp": "contains",
            "value_contains": ["SNR"],
        }}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0
        assert "value_contains" in details["matches"]

    def test_clauses_must_include_order_independent(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """clauses_must_include should match regardless of order."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "ASSOC", "cmp": "contains", "value": "47 Tuc"},
            {"field": "P0", "cmp": "lt", "value": 0.03},
        ]}}
        pattern = {"filters": {
            "op": "and",
            "clauses_must_include": [
                {"field": "P0", "cmp": "lt"},
                {"field": "ASSOC", "cmp": "contains", "value_contains": ["47", "Tuc"]},
            ],
        }}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0

    def test_clauses_must_include_partial(self, evaluator: ResponseEvaluator) -> None:
        """Missing one clause should give partial credit."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "P0", "cmp": "lt", "value": 0.03},
        ]}}
        pattern = {"filters": {
            "op": "and",
            "clauses_must_include": [
                {"field": "P0", "cmp": "lt"},
                {"field": "ASSOC", "cmp": "contains"},
            ],
        }}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert 0 < score < 1.0

    def test_select_fields_must_include(self, evaluator: ResponseEvaluator) -> None:
        """select_fields containing required fields should pass."""
        dsl = {"select_fields": ["JNAME", "P0", "DM"], "filters": None}
        pattern = {"select_fields_must_include": ["JNAME", "P0"]}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0

    def test_select_fields_none_passes(self, evaluator: ResponseEvaluator) -> None:
        """None select_fields (all fields) should pass any must_include check."""
        dsl = {"select_fields": None, "filters": None}
        pattern = {"select_fields_must_include": ["JNAME", "P0"]}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0

    def test_limit_range_in_range(self, evaluator: ResponseEvaluator) -> None:
        """Limit within range should pass."""
        dsl = {"select_fields": ["JNAME", "P0"], "limit": 5, "filters": None}
        pattern = {"select_fields_must_include": ["JNAME", "P0"], "limit_range": [5, 10]}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0
        assert "limit_range" in details["matches"]

    def test_limit_range_out_of_range(self, evaluator: ResponseEvaluator) -> None:
        """Limit outside range should fail that check."""
        dsl = {"select_fields": ["JNAME", "P0"], "limit": 100, "filters": None}
        pattern = {"select_fields_must_include": ["JNAME", "P0"], "limit_range": [5, 10]}

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score < 1.0
        assert any(m.get("check") == "limit_range" for m in details["mismatches"])

    def test_literal_path_patterns_still_work(
        self, evaluator: ResponseEvaluator
    ) -> None:
        """Existing literal-path patterns should still match correctly."""
        dsl = {"filters": {"op": "and", "clauses": [
            {"field": "P0", "cmp": "lt", "value": 0.01},
        ]}}
        pattern = {
            "filters.clauses[0].field": "P0",
            "filters.clauses[0].cmp": "lt",
            "filters.clauses[0].value": 0.01,
        }

        score, details = evaluator.evaluate_dsl_pattern(dsl, pattern)

        assert score == 1.0
