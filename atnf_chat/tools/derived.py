"""Derived parameter computation tools.

This module provides functions for computing derived pulsar parameters,
with preference for ATNF-native values when available.

Key principle: Always prefer ATNF-computed values (BSURF, EDOT, AGE)
when they exist in the catalogue, and document assumptions when
computing values ourselves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atnf_chat.core.schema import SchemaGroundingPack

logger = logging.getLogger(__name__)


# Physical constants for calculations
MOMENT_OF_INERTIA = 1.0e45  # g cm^2, canonical neutron star
NEUTRON_STAR_RADIUS = 1.0e6  # cm (10 km)
SPEED_OF_LIGHT = 2.998e10  # cm/s


@dataclass
class DerivedParameterResult:
    """Result from computing a derived parameter.

    Attributes:
        parameter: Name of the computed parameter
        values: Computed values as pandas Series
        source: Where values came from ("atnf_native" or "computed")
        formula: Formula used for computation
        assumptions: Physical assumptions made
        missing_count: Number of pulsars with insufficient data
        completeness: Fraction of valid results
    """

    parameter: str
    values: pd.Series
    source: str
    formula: str | None = None
    assumptions: dict[str, str] = field(default_factory=dict)
    missing_count: int = 0
    completeness: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding large data)."""
        return {
            "parameter": self.parameter,
            "source": self.source,
            "formula": self.formula,
            "assumptions": self.assumptions,
            "missing_count": self.missing_count,
            "completeness": self.completeness,
            "value_count": len(self.values) - self.missing_count,
        }

    def format_for_display(self) -> str:
        """Format as human-readable string."""
        valid = self.values.dropna()
        n_valid = len(valid)

        lines = [
            f"**{self.parameter}** (source: {self.source}, n={n_valid})",
        ]

        if n_valid > 0:
            lines.append(f"  Mean: {float(valid.mean()):.4g}")
            lines.append(f"  Median: {float(valid.median()):.4g}")
            lines.append(f"  Range: [{float(valid.min()):.4g}, {float(valid.max()):.4g}]")

        lines.append(f"  Completeness: {self.completeness:.1%}")

        if self.formula:
            lines.append(f"  Formula: {self.formula}")
        if self.assumptions:
            lines.append("  Assumptions: " + "; ".join(
                f"{k}={v}" for k, v in self.assumptions.items()
            ))

        return "\n".join(lines)


def compute_derived_parameter(
    df: pd.DataFrame,
    parameter: str,
    use_atnf_native: bool = True,
) -> DerivedParameterResult:
    """Compute a derived parameter for pulsars.

    This function follows the principle of preferring ATNF-native derived
    values when available. If the parameter is already in the catalogue
    (like BSURF, EDOT, AGE), those values are used. Only when necessary
    do we compute values ourselves, and we document all assumptions.

    Args:
        df: DataFrame with pulsar data
        parameter: Name of parameter to compute
        use_atnf_native: If True, prefer ATNF values when available

    Returns:
        DerivedParameterResult with values and metadata

    Example:
        >>> result = compute_derived_parameter(df, "BSURF")
        >>> print(f"Source: {result.source}")
        >>> print(f"Completeness: {result.completeness:.1%}")
    """
    param_upper = parameter.upper()
    schema = SchemaGroundingPack()
    param_info = schema.get_parameter(param_upper)

    # Check if ATNF provides this parameter natively
    if use_atnf_native and param_upper in df.columns:
        values = df[param_upper].copy()
        missing = values.isna().sum()

        return DerivedParameterResult(
            parameter=param_upper,
            values=values,
            source="atnf_native",
            formula=param_info.formula if param_info else "ATNF-computed",
            missing_count=missing,
            completeness=1.0 - (missing / len(values)) if len(values) > 0 else 0.0,
        )

    # Compute the parameter ourselves
    if param_upper == "BSURF":
        return _compute_bsurf(df)
    elif param_upper == "EDOT":
        return _compute_edot(df)
    elif param_upper in ("AGE", "CHAR_AGE", "TAU_C"):
        return _compute_characteristic_age(df)
    elif param_upper == "B_LC":
        return _compute_b_lc(df)
    elif param_upper == "VTRANS":
        return _compute_vtrans(df)
    else:
        raise ValueError(
            f"Unknown derived parameter: {parameter}. "
            f"Supported: BSURF, EDOT, AGE, B_LC, VTRANS"
        )


def _compute_bsurf(df: pd.DataFrame) -> DerivedParameterResult:
    """Compute surface magnetic field strength.

    Formula: B_surf = 3.2e19 * sqrt(P * Pdot) Gauss

    This assumes:
    - Magnetic dipole braking
    - Canonical moment of inertia (I = 1e45 g cm^2)
    - Canonical neutron star radius (R = 10 km)
    """
    if "P0" not in df.columns or "P1" not in df.columns:
        raise ValueError("BSURF requires P0 and P1 columns")

    P = df["P0"]
    Pdot = df["P1"]

    # Standard formula
    bsurf = 3.2e19 * np.sqrt(P * Pdot)

    missing = bsurf.isna().sum()

    return DerivedParameterResult(
        parameter="BSURF",
        values=bsurf,
        source="computed",
        formula="B_surf = 3.2e19 * sqrt(P0 * P1) Gauss",
        assumptions={
            "braking_mechanism": "magnetic dipole",
            "moment_of_inertia": "1.0e45 g cm^2",
            "neutron_star_radius": "10 km",
        },
        missing_count=missing,
        completeness=1.0 - (missing / len(bsurf)) if len(bsurf) > 0 else 0.0,
    )


def _compute_edot(df: pd.DataFrame) -> DerivedParameterResult:
    """Compute spin-down energy loss rate.

    Formula: Edot = 4 * pi^2 * I * Pdot / P^3 erg/s

    This assumes:
    - Canonical moment of inertia (I = 1e45 g cm^2)
    """
    # Prefer frequency-based if available
    if "F0" in df.columns and "F1" in df.columns:
        F = df["F0"]
        Fdot = df["F1"]

        # Edot = 4 * pi^2 * I * F^3 * |Fdot|
        edot = 4 * np.pi**2 * MOMENT_OF_INERTIA * F**3 * np.abs(Fdot)

        formula = "Edot = 4*pi^2 * I * F0^3 * |F1| erg/s"
    elif "P0" in df.columns and "P1" in df.columns:
        P = df["P0"]
        Pdot = df["P1"]

        # Edot = 4 * pi^2 * I * Pdot / P^3
        edot = 4 * np.pi**2 * MOMENT_OF_INERTIA * Pdot / P**3

        formula = "Edot = 4*pi^2 * I * P1 / P0^3 erg/s"
    else:
        raise ValueError("EDOT requires (F0, F1) or (P0, P1) columns")

    missing = edot.isna().sum()

    return DerivedParameterResult(
        parameter="EDOT",
        values=edot,
        source="computed",
        formula=formula,
        assumptions={
            "moment_of_inertia": "1.0e45 g cm^2",
        },
        missing_count=missing,
        completeness=1.0 - (missing / len(edot)) if len(edot) > 0 else 0.0,
    )


def _compute_characteristic_age(df: pd.DataFrame) -> DerivedParameterResult:
    """Compute characteristic age (spin-down age).

    Formula: tau_c = P / (2 * Pdot) seconds, converted to years

    This assumes:
    - Braking index n = 3 (magnetic dipole)
    - Initial period P0 << current period
    """
    if "P0" not in df.columns or "P1" not in df.columns:
        raise ValueError("Characteristic age requires P0 and P1 columns")

    P = df["P0"]
    Pdot = df["P1"]

    # tau in seconds
    tau_seconds = P / (2 * Pdot)

    # Convert to years
    seconds_per_year = 365.25 * 24 * 3600
    tau_years = tau_seconds / seconds_per_year

    missing = tau_years.isna().sum()

    return DerivedParameterResult(
        parameter="AGE",
        values=tau_years,
        source="computed",
        formula="tau_c = P0 / (2 * P1) [converted to years]",
        assumptions={
            "braking_index": "3 (magnetic dipole)",
            "initial_period": "P0_initial << P0_current",
        },
        missing_count=missing,
        completeness=1.0 - (missing / len(tau_years)) if len(tau_years) > 0 else 0.0,
    )


def _compute_b_lc(df: pd.DataFrame) -> DerivedParameterResult:
    """Compute magnetic field at light cylinder.

    Formula: B_LC = B_surf * (R_NS / R_LC)^3
    where R_LC = c * P / (2 * pi)
    """
    # First get surface field
    if "BSURF" in df.columns:
        bsurf = df["BSURF"]
    else:
        bsurf_result = _compute_bsurf(df)
        bsurf = bsurf_result.values

    if "P0" not in df.columns:
        raise ValueError("B_LC requires P0 column")

    P = df["P0"]

    # Light cylinder radius
    R_LC = SPEED_OF_LIGHT * P / (2 * np.pi)

    # B_LC = B_surf * (R_NS / R_LC)^3
    b_lc = bsurf * (NEUTRON_STAR_RADIUS / R_LC) ** 3

    missing = b_lc.isna().sum()

    return DerivedParameterResult(
        parameter="B_LC",
        values=b_lc,
        source="computed",
        formula="B_LC = B_surf * (R_NS / R_LC)^3, R_LC = c*P/(2*pi)",
        assumptions={
            "neutron_star_radius": "10 km",
        },
        missing_count=missing,
        completeness=1.0 - (missing / len(b_lc)) if len(b_lc) > 0 else 0.0,
    )


def _compute_vtrans(df: pd.DataFrame) -> DerivedParameterResult:
    """Compute transverse velocity.

    Formula: V_trans = 4.74 * mu * D km/s
    where mu is proper motion in mas/yr and D is distance in kpc

    4.74 is the conversion factor from mas/yr * kpc to km/s
    """
    # Check for required columns
    has_pmtot = "PMTOT" in df.columns
    has_pm_components = "PMRA" in df.columns and "PMDEC" in df.columns
    has_dist = "DIST" in df.columns

    if not has_dist:
        raise ValueError("VTRANS requires DIST column")

    if has_pmtot:
        pm = df["PMTOT"]
    elif has_pm_components:
        pmra = df["PMRA"]
        pmdec = df["PMDEC"]
        pm = np.sqrt(pmra**2 + pmdec**2)
    else:
        raise ValueError("VTRANS requires PMTOT or (PMRA, PMDEC) columns")

    dist = df["DIST"]

    # V_trans = 4.74 * pm * D
    vtrans = 4.74 * pm * dist

    missing = vtrans.isna().sum()

    return DerivedParameterResult(
        parameter="VTRANS",
        values=vtrans,
        source="computed",
        formula="V_trans = 4.74 * mu * D km/s",
        assumptions={
            "proper_motion_unit": "mas/yr",
            "distance_unit": "kpc",
            "conversion_factor": "4.74 km/s per (mas/yr * kpc)",
        },
        missing_count=missing,
        completeness=1.0 - (missing / len(vtrans)) if len(vtrans) > 0 else 0.0,
    )


def get_available_derived_parameters() -> dict[str, dict[str, Any]]:
    """Get information about available derived parameters.

    Returns:
        Dictionary mapping parameter names to their metadata
    """
    return {
        "BSURF": {
            "description": "Surface magnetic field strength",
            "unit": "Gauss",
            "requires": ["P0", "P1"],
            "atnf_native": True,
        },
        "EDOT": {
            "description": "Spin-down energy loss rate",
            "unit": "erg/s",
            "requires": ["P0", "P1"] ,
            "atnf_native": True,
        },
        "AGE": {
            "description": "Characteristic age (spin-down age)",
            "unit": "years",
            "requires": ["P0", "P1"],
            "atnf_native": True,
        },
        "B_LC": {
            "description": "Magnetic field at light cylinder",
            "unit": "Gauss",
            "requires": ["BSURF", "P0"],
            "atnf_native": True,
        },
        "VTRANS": {
            "description": "Transverse velocity",
            "unit": "km/s",
            "requires": ["PMTOT or (PMRA, PMDEC)", "DIST"],
            "atnf_native": True,
        },
    }
