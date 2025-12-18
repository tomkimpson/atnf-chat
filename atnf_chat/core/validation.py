"""Result validation and scientific safety for ATNF-Chat.

This module provides validation and safety checks for query results,
ensuring scientific rigor and helping users avoid common pitfalls.

Features:
- High missingness detection and warnings
- Selection effect identification
- Empty result suggestions
- Data quality assessments
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from atnf_chat.core.catalogue import QueryProvenance
    from atnf_chat.core.dsl import QueryDSL


class WarningType(str, Enum):
    """Types of warnings that can be generated."""

    HIGH_MISSINGNESS = "high_missingness"
    SELECTION_EFFECT = "selection_effect"
    EPOCH_RANGE = "epoch_range"
    EMPTY_RESULT = "empty_result"
    LARGE_RESULT = "large_result"
    VALUE_RANGE = "value_range"
    DERIVED_ASSUMPTION = "derived_assumption"


class WarningSeverity(str, Enum):
    """Severity levels for warnings."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationWarning:
    """A single validation warning.

    Attributes:
        type: Category of the warning
        severity: How serious the warning is
        field: Parameter field involved (if applicable)
        message: Human-readable warning message
        suggestion: Suggested action to resolve
        details: Additional context
    """

    type: WarningType
    severity: WarningSeverity
    message: str
    field: str | None = None
    suggestion: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Complete validation result for a query.

    Attributes:
        is_valid: Whether the result passes basic validation
        is_safe: Whether the result is scientifically safe to use
        warnings: List of warnings generated
        suggestions: List of suggested improvements
    """

    is_valid: bool
    is_safe: bool
    warnings: list[ValidationWarning] = dataclass_field(default_factory=list)
    suggestions: list[str] = dataclass_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "is_safe": self.is_safe,
            "warnings": [w.to_dict() for w in self.warnings],
            "suggestions": self.suggestions,
            "warning_count": len(self.warnings),
            "critical_count": len(
                [w for w in self.warnings if w.severity == WarningSeverity.CRITICAL]
            ),
        }

    def get_warnings_by_type(self, warn_type: WarningType) -> list[ValidationWarning]:
        """Get all warnings of a specific type."""
        return [w for w in self.warnings if w.type == warn_type]

    def has_critical_warnings(self) -> bool:
        """Check if there are any critical warnings."""
        return any(w.severity == WarningSeverity.CRITICAL for w in self.warnings)

    def format_for_llm(self) -> str:
        """Format validation result for LLM response generation."""
        if not self.warnings:
            return ""

        lines = ["**Data Quality Notes:**"]
        for warning in self.warnings:
            icon = "⚠️" if warning.severity == WarningSeverity.WARNING else "ℹ️"
            if warning.severity == WarningSeverity.CRITICAL:
                icon = "🚨"
            lines.append(f"- {icon} {warning.message}")
            if warning.suggestion:
                lines.append(f"  *Suggestion: {warning.suggestion}*")

        return "\n".join(lines)


class ResultValidator:
    """Validates query results for scientific safety.

    This class checks results for common issues that could affect
    scientific analyses, such as:
    - High rates of missing data
    - Selection effects from filtering
    - Epoch differences in timing parameters
    - Unusual value ranges

    Example:
        >>> validator = ResultValidator()
        >>> validation = validator.validate(results_df, provenance, query_dsl)
        >>> if not validation.is_safe:
        ...     print(validation.format_for_llm())
    """

    # Thresholds for various checks
    HIGH_MISSINGNESS_THRESHOLD = 0.5  # 50% missing triggers warning
    CRITICAL_MISSINGNESS_THRESHOLD = 0.8  # 80% missing is critical
    LARGE_RESULT_THRESHOLD = 1000
    EPOCH_RANGE_WARNING_YEARS = 20

    # Parameters that commonly have selection effects
    SELECTION_EFFECT_PARAMS = {"ASSOC", "TYPE", "SURVEY", "BINARY", "BINCOMP"}

    # Parameters where epoch matters
    EPOCH_SENSITIVE_PARAMS = {"RAJD", "DECJD", "PMRA", "PMDEC", "PX"}

    def __init__(
        self,
        missingness_threshold: float = 0.5,
        critical_threshold: float = 0.8,
    ) -> None:
        """Initialize the validator.

        Args:
            missingness_threshold: Fraction of missing data to trigger warning
            critical_threshold: Fraction of missing data for critical warning
        """
        self.missingness_threshold = missingness_threshold
        self.critical_threshold = critical_threshold

    def validate(
        self,
        df: pd.DataFrame,
        provenance: QueryProvenance,
        query_dsl: QueryDSL | None = None,
    ) -> ValidationResult:
        """Validate query results.

        Args:
            df: Query results DataFrame
            provenance: Query provenance information
            query_dsl: Original query (optional, for context)

        Returns:
            ValidationResult with warnings and suggestions
        """
        warnings: list[ValidationWarning] = []
        suggestions: list[str] = []

        # Check for empty results
        if len(df) == 0:
            warnings.append(self._empty_result_warning(query_dsl))
            suggestions.extend(self._suggest_relaxations(query_dsl))
            return ValidationResult(
                is_valid=True,
                is_safe=False,
                warnings=warnings,
                suggestions=suggestions,
            )

        # Check for large results
        if len(df) > self.LARGE_RESULT_THRESHOLD:
            warnings.append(self._large_result_warning(len(df)))
            suggestions.append(
                "Consider adding filters to focus on a specific subset of pulsars"
            )

        # Check missingness for each field
        for field, completeness in provenance.completeness.items():
            missingness = 1.0 - completeness
            if missingness >= self.critical_threshold:
                warnings.append(
                    self._missingness_warning(
                        field, missingness, WarningSeverity.CRITICAL
                    )
                )
            elif missingness >= self.missingness_threshold:
                warnings.append(
                    self._missingness_warning(
                        field, missingness, WarningSeverity.WARNING
                    )
                )

        # Check for selection effects
        if query_dsl and query_dsl.filters:
            selection_warnings = self._check_selection_effects(query_dsl)
            warnings.extend(selection_warnings)

        # Check epoch range for position-sensitive parameters
        epoch_warnings = self._check_epoch_range(df)
        warnings.extend(epoch_warnings)

        # Determine if result is scientifically safe
        is_safe = not any(
            w.severity == WarningSeverity.CRITICAL
            and w.type == WarningType.HIGH_MISSINGNESS
            for w in warnings
        )

        return ValidationResult(
            is_valid=True,
            is_safe=is_safe,
            warnings=warnings,
            suggestions=suggestions,
        )

    def _empty_result_warning(self, query_dsl: QueryDSL | None) -> ValidationWarning:
        """Generate warning for empty results."""
        return ValidationWarning(
            type=WarningType.EMPTY_RESULT,
            severity=WarningSeverity.WARNING,
            message="No pulsars match your query criteria.",
            suggestion="Try relaxing your filter conditions.",
            details={"query": query_dsl.model_dump() if query_dsl else None},
        )

    def _large_result_warning(self, count: int) -> ValidationWarning:
        """Generate warning for large result sets."""
        return ValidationWarning(
            type=WarningType.LARGE_RESULT,
            severity=WarningSeverity.INFO,
            message=f"Query returned {count} pulsars, which may be slow to process.",
            suggestion="Consider adding filters for more focused analysis.",
            details={"result_count": count},
        )

    def _missingness_warning(
        self,
        field: str,
        missingness: float,
        severity: WarningSeverity,
    ) -> ValidationWarning:
        """Generate warning for high missingness."""
        pct = missingness * 100
        return ValidationWarning(
            type=WarningType.HIGH_MISSINGNESS,
            severity=severity,
            field=field,
            message=f"{field} is missing for {pct:.0f}% of results.",
            suggestion=f"Derived calculations using {field} will be incomplete.",
            details={"missingness": missingness, "completeness": 1.0 - missingness},
        )

    def _check_selection_effects(
        self, query_dsl: QueryDSL
    ) -> list[ValidationWarning]:
        """Check for potential selection effects in the query."""
        warnings = []

        if query_dsl.filters is None:
            return warnings

        # Get all fields used in filters
        filter_fields = self._extract_filter_fields(query_dsl.filters)

        for field in filter_fields:
            if field in self.SELECTION_EFFECT_PARAMS:
                if field == "ASSOC":
                    warnings.append(
                        ValidationWarning(
                            type=WarningType.SELECTION_EFFECT,
                            severity=WarningSeverity.INFO,
                            field=field,
                            message=(
                                "Association filtering may introduce selection bias. "
                                "ASSOC uses heterogeneous naming conventions."
                            ),
                            suggestion=(
                                "Verify matches manually for critical analyses. "
                                "Different catalogues use different association names."
                            ),
                        )
                    )
                elif field == "BINARY":
                    warnings.append(
                        ValidationWarning(
                            type=WarningType.SELECTION_EFFECT,
                            severity=WarningSeverity.INFO,
                            field=field,
                            message=(
                                "Binary status may be incomplete. "
                                "Some pulsars may have undetected companions."
                            ),
                        )
                    )

        return warnings

    def _extract_filter_fields(self, filter_group: Any) -> set[str]:
        """Extract all field names from a filter group."""
        from atnf_chat.core.dsl import FilterClause, FilterGroup

        fields = set()

        for clause in filter_group.clauses:
            if isinstance(clause, FilterClause):
                fields.add(clause.field)
            elif isinstance(clause, FilterGroup):
                fields.update(self._extract_filter_fields(clause))

        return fields

    def _check_epoch_range(self, df: pd.DataFrame) -> list[ValidationWarning]:
        """Check for large epoch ranges in timing parameters."""
        warnings = []

        if "POSEPOCH" not in df.columns:
            return warnings

        epochs = df["POSEPOCH"].dropna()
        if len(epochs) < 2:
            return warnings

        epoch_range = epochs.max() - epochs.min()
        # Convert MJD days to years
        epoch_range_years = epoch_range / 365.25

        if epoch_range_years > self.EPOCH_RANGE_WARNING_YEARS:
            warnings.append(
                ValidationWarning(
                    type=WarningType.EPOCH_RANGE,
                    severity=WarningSeverity.INFO,
                    message=(
                        f"Position epochs span {epoch_range_years:.0f} years. "
                        "Proper motion may affect coordinate comparisons."
                    ),
                    suggestion=(
                        "Consider correcting positions to a common epoch "
                        "for spatial analyses."
                    ),
                    details={
                        "epoch_range_years": epoch_range_years,
                        "min_epoch": float(epochs.min()),
                        "max_epoch": float(epochs.max()),
                    },
                )
            )

        return warnings

    def _suggest_relaxations(self, query_dsl: QueryDSL | None) -> list[str]:
        """Suggest how to relax an overly restrictive query."""
        if query_dsl is None or query_dsl.filters is None:
            return ["Try a query without filters to explore the full catalogue."]

        suggestions = []
        filter_fields = self._extract_filter_fields(query_dsl.filters)

        # Generic suggestions based on common filter types
        if "P0" in filter_fields:
            suggestions.append(
                "Increase period threshold (e.g., P0 < 0.1s instead of P0 < 0.03s)"
            )
        if "ASSOC" in filter_fields:
            suggestions.append(
                "Try a broader association search or remove the ASSOC filter"
            )
        if "PB" in filter_fields or "BINARY" in filter_fields:
            suggestions.append(
                "Consider including isolated pulsars by removing binary filters"
            )
        if "DM" in filter_fields:
            suggestions.append("Expand DM range to include more distant pulsars")

        if not suggestions:
            suggestions.append("Try relaxing one or more filter conditions")

        return suggestions

    def validate_derived_calculation(
        self,
        param_name: str,
        source: str,
        assumptions: dict[str, Any],
    ) -> ValidationWarning | None:
        """Validate a derived parameter calculation.

        Args:
            param_name: Name of derived parameter
            source: Source of the value ("atnf_native" or "computed")
            assumptions: Physical assumptions used in computation

        Returns:
            Warning if assumptions should be noted, None otherwise
        """
        if source == "atnf_native":
            return None

        # Computed values should note assumptions
        assumption_text = ", ".join(f"{k}={v}" for k, v in assumptions.items())

        return ValidationWarning(
            type=WarningType.DERIVED_ASSUMPTION,
            severity=WarningSeverity.INFO,
            field=param_name,
            message=f"{param_name} computed using standard assumptions.",
            suggestion=f"Calculation assumes: {assumption_text}",
            details={"assumptions": assumptions},
        )
