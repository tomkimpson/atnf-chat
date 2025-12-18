"""Tests for the validation module."""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from atnf_chat.core.validation import (
    ResultValidator,
    ValidationResult,
    ValidationWarning,
    WarningType,
    WarningSeverity,
)
from atnf_chat.core.dsl import QueryDSL, FilterGroup, FilterClause, ComparisonOp


class TestValidationWarning:
    """Tests for ValidationWarning dataclass."""

    def test_create_warning(self) -> None:
        """Test creating a warning."""
        warning = ValidationWarning(
            type=WarningType.HIGH_MISSINGNESS,
            severity=WarningSeverity.WARNING,
            message="Test warning",
            field="P0",
        )
        assert warning.type == WarningType.HIGH_MISSINGNESS
        assert warning.field == "P0"

    def test_warning_to_dict(self) -> None:
        """Test converting warning to dictionary."""
        warning = ValidationWarning(
            type=WarningType.EMPTY_RESULT,
            severity=WarningSeverity.WARNING,
            message="No results",
            suggestion="Try relaxing filters",
        )
        d = warning.to_dict()
        assert d["type"] == "empty_result"
        assert d["severity"] == "warning"
        assert d["suggestion"] == "Try relaxing filters"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a validation result."""
        result = ValidationResult(is_valid=True, is_safe=True)
        assert result.is_valid
        assert result.is_safe
        assert len(result.warnings) == 0

    def test_result_with_warnings(self) -> None:
        """Test result with warnings."""
        warnings = [
            ValidationWarning(
                type=WarningType.HIGH_MISSINGNESS,
                severity=WarningSeverity.WARNING,
                message="Test",
            ),
            ValidationWarning(
                type=WarningType.SELECTION_EFFECT,
                severity=WarningSeverity.INFO,
                message="Info",
            ),
        ]
        result = ValidationResult(is_valid=True, is_safe=True, warnings=warnings)
        assert len(result.warnings) == 2

    def test_get_warnings_by_type(self) -> None:
        """Test filtering warnings by type."""
        warnings = [
            ValidationWarning(
                type=WarningType.HIGH_MISSINGNESS,
                severity=WarningSeverity.WARNING,
                message="Missing 1",
            ),
            ValidationWarning(
                type=WarningType.SELECTION_EFFECT,
                severity=WarningSeverity.INFO,
                message="Selection",
            ),
            ValidationWarning(
                type=WarningType.HIGH_MISSINGNESS,
                severity=WarningSeverity.CRITICAL,
                message="Missing 2",
            ),
        ]
        result = ValidationResult(is_valid=True, is_safe=False, warnings=warnings)

        missingness_warnings = result.get_warnings_by_type(WarningType.HIGH_MISSINGNESS)
        assert len(missingness_warnings) == 2

    def test_has_critical_warnings(self) -> None:
        """Test detecting critical warnings."""
        warnings_no_critical = [
            ValidationWarning(
                type=WarningType.HIGH_MISSINGNESS,
                severity=WarningSeverity.WARNING,
                message="Test",
            ),
        ]
        result1 = ValidationResult(is_valid=True, is_safe=True, warnings=warnings_no_critical)
        assert not result1.has_critical_warnings()

        warnings_critical = [
            ValidationWarning(
                type=WarningType.HIGH_MISSINGNESS,
                severity=WarningSeverity.CRITICAL,
                message="Critical",
            ),
        ]
        result2 = ValidationResult(is_valid=True, is_safe=False, warnings=warnings_critical)
        assert result2.has_critical_warnings()

    def test_format_for_llm(self) -> None:
        """Test formatting result for LLM."""
        warnings = [
            ValidationWarning(
                type=WarningType.HIGH_MISSINGNESS,
                severity=WarningSeverity.WARNING,
                message="ECC is missing for 60% of results",
                suggestion="Derived calculations may be incomplete",
            ),
        ]
        result = ValidationResult(is_valid=True, is_safe=True, warnings=warnings)

        formatted = result.format_for_llm()
        assert "Data Quality Notes" in formatted
        assert "ECC is missing" in formatted
        assert "60%" in formatted


class TestResultValidator:
    """Tests for ResultValidator class."""

    @pytest.fixture
    def validator(self) -> ResultValidator:
        """Create a validator instance."""
        return ResultValidator()

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame."""
        return pd.DataFrame({
            "JNAME": ["J0001", "J0002", "J0003", "J0004", "J0005"],
            "P0": [0.01, 0.02, 0.03, 0.04, 0.05],
            "DM": [10.0, 20.0, None, 40.0, 50.0],  # 20% missing
            "ECC": [None, None, None, 0.1, None],  # 80% missing
        })

    @pytest.fixture
    def mock_provenance(self) -> MagicMock:
        """Create mock provenance."""
        prov = MagicMock()
        prov.completeness = {
            "JNAME": 1.0,
            "P0": 1.0,
            "DM": 0.8,  # 20% missing
            "ECC": 0.2,  # 80% missing
        }
        prov.null_counts = {
            "JNAME": 0,
            "P0": 0,
            "DM": 1,
            "ECC": 4,
        }
        return prov

    def test_validate_good_data(
        self, validator: ResultValidator, sample_df: pd.DataFrame
    ) -> None:
        """Test validation of good quality data."""
        prov = MagicMock()
        prov.completeness = {"JNAME": 1.0, "P0": 1.0}
        prov.null_counts = {"JNAME": 0, "P0": 0}

        result = validator.validate(sample_df[["JNAME", "P0"]], prov)

        assert result.is_valid
        assert result.is_safe
        assert len(result.warnings) == 0

    def test_validate_high_missingness(
        self,
        validator: ResultValidator,
        sample_df: pd.DataFrame,
        mock_provenance: MagicMock,
    ) -> None:
        """Test detection of high missingness."""
        result = validator.validate(sample_df, mock_provenance)

        # Should have warning for ECC (80% missing > 50% threshold)
        missingness_warnings = result.get_warnings_by_type(WarningType.HIGH_MISSINGNESS)
        assert len(missingness_warnings) >= 1

        ecc_warning = next(
            (w for w in missingness_warnings if w.field == "ECC"), None
        )
        assert ecc_warning is not None
        assert ecc_warning.severity == WarningSeverity.CRITICAL  # 80% > critical threshold

    def test_validate_empty_result(self, validator: ResultValidator) -> None:
        """Test validation of empty results."""
        empty_df = pd.DataFrame()
        prov = MagicMock()
        prov.completeness = {}
        prov.null_counts = {}

        query = QueryDSL(
            filters=FilterGroup(
                op="and",
                clauses=[FilterClause(field="P0", cmp=ComparisonOp.LT, value=0.001)],
            )
        )

        result = validator.validate(empty_df, prov, query)

        assert result.is_valid
        assert not result.is_safe
        empty_warnings = result.get_warnings_by_type(WarningType.EMPTY_RESULT)
        assert len(empty_warnings) == 1
        assert len(result.suggestions) > 0

    def test_validate_large_result(self, validator: ResultValidator) -> None:
        """Test warning for large result sets."""
        large_df = pd.DataFrame({"JNAME": [f"J{i:04d}" for i in range(1500)]})
        prov = MagicMock()
        prov.completeness = {"JNAME": 1.0}
        prov.null_counts = {"JNAME": 0}

        result = validator.validate(large_df, prov)

        large_warnings = result.get_warnings_by_type(WarningType.LARGE_RESULT)
        assert len(large_warnings) == 1
        assert "1500" in large_warnings[0].message

    def test_validate_selection_effects(
        self, validator: ResultValidator, sample_df: pd.DataFrame
    ) -> None:
        """Test detection of selection effects."""
        prov = MagicMock()
        prov.completeness = {"JNAME": 1.0}
        prov.null_counts = {"JNAME": 0}

        query = QueryDSL(
            filters=FilterGroup(
                op="and",
                clauses=[
                    FilterClause(field="ASSOC", cmp=ComparisonOp.CONTAINS, value="GC")
                ],
            )
        )

        result = validator.validate(sample_df[["JNAME"]], prov, query)

        selection_warnings = result.get_warnings_by_type(WarningType.SELECTION_EFFECT)
        assert len(selection_warnings) >= 1
        assert "ASSOC" in selection_warnings[0].field

    def test_validate_epoch_range(self, validator: ResultValidator) -> None:
        """Test detection of large epoch ranges."""
        df = pd.DataFrame({
            "JNAME": ["J0001", "J0002", "J0003"],
            "POSEPOCH": [50000.0, 55000.0, 60000.0],  # ~27 year span
        })
        prov = MagicMock()
        prov.completeness = {"JNAME": 1.0, "POSEPOCH": 1.0}
        prov.null_counts = {"JNAME": 0, "POSEPOCH": 0}

        result = validator.validate(df, prov)

        epoch_warnings = result.get_warnings_by_type(WarningType.EPOCH_RANGE)
        assert len(epoch_warnings) == 1
        assert "years" in epoch_warnings[0].message

    def test_validate_derived_calculation(self, validator: ResultValidator) -> None:
        """Test validation of derived calculations."""
        warning = validator.validate_derived_calculation(
            param_name="BSURF",
            source="computed",
            assumptions={"moment_of_inertia": "1e45 g cm^2"},
        )

        assert warning is not None
        assert warning.type == WarningType.DERIVED_ASSUMPTION
        assert "moment_of_inertia" in warning.suggestion

    def test_validate_derived_atnf_native(self, validator: ResultValidator) -> None:
        """Test that ATNF-native values don't generate warnings."""
        warning = validator.validate_derived_calculation(
            param_name="BSURF",
            source="atnf_native",
            assumptions={},
        )

        assert warning is None

    def test_custom_thresholds(self, sample_df: pd.DataFrame) -> None:
        """Test validator with custom thresholds."""
        validator = ResultValidator(
            missingness_threshold=0.1,  # Very strict
            critical_threshold=0.5,
        )
        prov = MagicMock()
        prov.completeness = {"DM": 0.8}  # 20% missing
        prov.null_counts = {"DM": 1}

        result = validator.validate(sample_df[["DM"]], prov)

        # Should flag DM as missing (20% > 10% threshold)
        warnings = result.get_warnings_by_type(WarningType.HIGH_MISSINGNESS)
        assert len(warnings) == 1
