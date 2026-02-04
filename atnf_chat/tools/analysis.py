"""Statistical analysis tools for pulsar data.

This module provides statistical analysis functions including:
- Summary statistics
- Correlation analysis
- Distribution analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatisticalSummary:
    """Summary statistics for a parameter.

    Attributes:
        parameter: Parameter name
        count: Number of non-null values
        mean: Arithmetic mean
        median: Median value
        std: Standard deviation
        min: Minimum value
        max: Maximum value
        q25: 25th percentile
        q75: 75th percentile
        iqr: Interquartile range
        skewness: Distribution skewness
        kurtosis: Distribution kurtosis
    """

    parameter: str
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    skewness: float | None = None
    kurtosis: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "q25": self.q25,
            "q75": self.q75,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }

    def format_for_display(self) -> str:
        """Format as human-readable string."""
        lines = [
            f"**{self.parameter}** (n={self.count})",
            f"  Mean: {self.mean:.4g}",
            f"  Median: {self.median:.4g}",
            f"  Std Dev: {self.std:.4g}",
            f"  Range: [{self.min:.4g}, {self.max:.4g}]",
            f"  IQR: [{self.q25:.4g}, {self.q75:.4g}]",
        ]
        return "\n".join(lines)


@dataclass
class CorrelationResult:
    """Result from correlation analysis.

    Attributes:
        param_x: First parameter name
        param_y: Second parameter name
        n_points: Number of data points used
        pearson_r: Pearson correlation coefficient
        pearson_p: Pearson p-value
        spearman_r: Spearman correlation coefficient
        spearman_p: Spearman p-value
        interpretation: Human-readable interpretation
    """

    param_x: str
    param_y: str
    n_points: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "param_x": self.param_x,
            "param_y": self.param_y,
            "n_points": self.n_points,
            "pearson": {
                "r": self.pearson_r,
                "p_value": self.pearson_p,
            },
            "spearman": {
                "r": self.spearman_r,
                "p_value": self.spearman_p,
            },
            "interpretation": self.interpretation,
        }

    def format_for_display(self) -> str:
        """Format as human-readable string."""
        lines = [
            f"**Correlation: {self.param_x} vs {self.param_y}** (n={self.n_points})",
            f"  Pearson r = {self.pearson_r:.3f} (p = {self.pearson_p:.2e})",
            f"  Spearman ρ = {self.spearman_r:.3f} (p = {self.spearman_p:.2e})",
        ]
        if self.interpretation:
            lines.append(f"  {self.interpretation}")
        return "\n".join(lines)


@dataclass
class AnalysisResult:
    """Complete result from statistical analysis.

    Attributes:
        success: Whether analysis completed successfully
        summaries: List of statistical summaries
        correlations: List of correlation results
        error: Error message if failed
    """

    success: bool
    summaries: list[StatisticalSummary] = field(default_factory=list)
    correlations: list[CorrelationResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "summaries": [s.to_dict() for s in self.summaries],
            "correlations": [c.to_dict() for c in self.correlations],
            "error": self.error,
        }

    def format_for_display(self) -> str:
        """Format as human-readable string."""
        if not self.success:
            return f"**Analysis Failed:** {self.error}"

        parts: list[str] = []
        for s in self.summaries:
            parts.append(s.format_for_display())
        for c in self.correlations:
            parts.append(c.format_for_display())

        return "\n\n".join(parts) if parts else "No analysis results."


def statistical_analysis(
    df: pd.DataFrame,
    parameters: list[str] | None = None,
) -> AnalysisResult:
    """Compute statistical summaries for parameters.

    Args:
        df: DataFrame with pulsar data
        parameters: List of parameter names (None = all numeric)

    Returns:
        AnalysisResult with summaries

    Example:
        >>> result = statistical_analysis(df, ["P0", "DM", "BSURF"])
        >>> for summary in result.summaries:
        ...     print(summary.format_for_display())
    """
    if parameters is None:
        # Use all numeric columns
        parameters = df.select_dtypes(include=[np.number]).columns.tolist()

    summaries = []

    for param in parameters:
        if param not in df.columns:
            continue

        data = df[param].dropna()

        if len(data) < 2:
            continue

        try:
            summary = StatisticalSummary(
                parameter=param,
                count=len(data),
                mean=float(data.mean()),
                median=float(data.median()),
                std=float(data.std()),
                min=float(data.min()),
                max=float(data.max()),
                q25=float(data.quantile(0.25)),
                q75=float(data.quantile(0.75)),
                iqr=float(data.quantile(0.75) - data.quantile(0.25)),
                skewness=float(stats.skew(data)) if len(data) >= 3 else None,
                kurtosis=float(stats.kurtosis(data)) if len(data) >= 4 else None,
            )
            summaries.append(summary)
        except Exception as e:
            # Skip parameters that cause issues
            continue

    return AnalysisResult(success=True, summaries=summaries)


def correlation_analysis(
    df: pd.DataFrame,
    param_x: str,
    param_y: str,
    use_log: bool = False,
) -> AnalysisResult:
    """Compute correlation between two parameters.

    Args:
        df: DataFrame with pulsar data
        param_x: First parameter name
        param_y: Second parameter name
        use_log: If True, use log10 of values (for power-law relationships)

    Returns:
        AnalysisResult with correlation

    Example:
        >>> result = correlation_analysis(df, "P0", "BSURF", use_log=True)
        >>> print(result.correlations[0].format_for_display())
    """
    if param_x not in df.columns:
        return AnalysisResult(
            success=False, error=f"Parameter {param_x} not found in data"
        )
    if param_y not in df.columns:
        return AnalysisResult(
            success=False, error=f"Parameter {param_y} not found in data"
        )

    # Get clean data (both values present)
    clean_df = df[[param_x, param_y]].dropna()

    if len(clean_df) < 3:
        return AnalysisResult(
            success=False,
            error=f"Insufficient data for correlation (need at least 3 points, got {len(clean_df)})",
        )

    x = clean_df[param_x].values
    y = clean_df[param_y].values

    # Apply log transform if requested
    if use_log:
        # Filter positive values only
        mask = (x > 0) & (y > 0)
        if mask.sum() < 3:
            return AnalysisResult(
                success=False,
                error="Insufficient positive values for log correlation",
            )
        x = np.log10(x[mask])
        y = np.log10(y[mask])
        n_points = mask.sum()
    else:
        n_points = len(x)

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # Generate interpretation
    interpretation = _interpret_correlation(pearson_r, pearson_p, spearman_r)

    result = CorrelationResult(
        param_x=param_x + (" (log)" if use_log else ""),
        param_y=param_y + (" (log)" if use_log else ""),
        n_points=n_points,
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
        interpretation=interpretation,
    )

    return AnalysisResult(success=True, correlations=[result])


def _interpret_correlation(
    pearson_r: float,
    pearson_p: float,
    spearman_r: float,
) -> str:
    """Generate human-readable interpretation of correlation."""
    abs_r = abs(pearson_r)

    # Strength
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    # Direction
    direction = "positive" if pearson_r > 0 else "negative"

    # Significance
    if pearson_p < 0.001:
        significance = "highly significant (p < 0.001)"
    elif pearson_p < 0.01:
        significance = "significant (p < 0.01)"
    elif pearson_p < 0.05:
        significance = "marginally significant (p < 0.05)"
    else:
        significance = "not statistically significant"

    # Linearity check (compare Pearson and Spearman)
    linearity = ""
    if abs(spearman_r) > abs(pearson_r) + 0.1:
        linearity = " The relationship may be non-linear."

    return f"{strength.capitalize()} {direction} correlation, {significance}.{linearity}"


def multi_correlation_analysis(
    df: pd.DataFrame,
    parameters: list[str],
    use_log: bool = False,
) -> AnalysisResult:
    """Compute correlations between multiple parameters.

    Args:
        df: DataFrame with pulsar data
        parameters: List of parameter names
        use_log: If True, use log10 of values

    Returns:
        AnalysisResult with all pairwise correlations
    """
    correlations = []

    for i, param_x in enumerate(parameters):
        for param_y in parameters[i + 1 :]:
            result = correlation_analysis(df, param_x, param_y, use_log=use_log)
            if result.success and result.correlations:
                correlations.extend(result.correlations)

    return AnalysisResult(success=True, correlations=correlations)


def compare_groups(
    df: pd.DataFrame,
    parameter: str,
    group_column: str,
) -> dict[str, Any]:
    """Compare a parameter between groups.

    Args:
        df: DataFrame with pulsar data
        parameter: Parameter to compare
        group_column: Column defining groups

    Returns:
        Dictionary with comparison results
    """
    if parameter not in df.columns or group_column not in df.columns:
        return {"success": False, "error": "Required columns not found"}

    groups = df.groupby(group_column)[parameter]

    group_stats = {}
    for name, group in groups:
        data = group.dropna()
        if len(data) > 0:
            group_stats[str(name)] = {
                "count": len(data),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
            }

    # Perform statistical test if we have exactly 2 groups
    group_names = list(group_stats.keys())
    test_result = None

    if len(group_names) == 2:
        g1 = groups.get_group(group_names[0]).dropna()
        g2 = groups.get_group(group_names[1]).dropna()

        if len(g1) >= 3 and len(g2) >= 3:
            # Mann-Whitney U test (non-parametric)
            stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_result = {
                "test": "Mann-Whitney U",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }

    return {
        "success": True,
        "parameter": parameter,
        "group_column": group_column,
        "group_stats": group_stats,
        "test_result": test_result,
    }
