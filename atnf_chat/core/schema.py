"""Schema Grounding Pack for ATNF Pulsar Catalogue parameters.

This module provides canonical parameter definitions, human-readable aliases,
and unit handling for the ATNF Pulsar Catalogue. Parameter definitions are
based on the official ATNF documentation.

Reference: https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from astropy import units as u


class ParameterType(str, Enum):
    """Classification of parameter types in the catalogue."""

    MEASURED = "measured"  # Directly measured from observations
    DERIVED_ATNF = "derived_atnf"  # Derived by ATNF, included in catalogue
    DERIVED_CUSTOM = "derived_custom"  # Must be computed by us
    CATEGORICAL = "categorical"  # Non-numeric (e.g., binary type, association)
    IDENTIFIER = "identifier"  # Pulsar names and identifiers


class ParameterCategory(str, Enum):
    """Broad categories of pulsar parameters."""

    IDENTIFICATION = "identification"
    TIMING = "timing"
    ASTROMETRIC = "astrometric"
    BINARY = "binary"
    DERIVED = "derived"
    ASSOCIATION = "association"
    SURVEY = "survey"


@dataclass(frozen=True)
class ParameterDefinition:
    """Definition of a single catalogue parameter."""

    code: str  # Official ATNF parameter code
    description: str
    unit: str | None  # Unit string (None for dimensionless/categorical)
    param_type: ParameterType
    category: ParameterCategory
    typical_range: tuple[float, float] | None = None  # (min, max) for validation
    related_params: tuple[str, ...] = field(default_factory=tuple)
    formula: str | None = None  # For derived parameters
    notes: str | None = None


class SchemaGroundingPack:
    """ATNF Pulsar Catalogue schema definitions and mappings.

    This class provides:
    - Canonical parameter definitions from ATNF documentation
    - Human-readable aliases for natural language processing
    - Unit registry using astropy.units
    - Validation helpers for query construction

    Example:
        >>> schema = SchemaGroundingPack()
        >>> schema.resolve_alias("spin period")
        'P0'
        >>> schema.get_parameter("P0").description
        'Barycentric period of the pulsar'
    """

    # Canonical ATNF parameter definitions
    # Based on: https://www.atnf.csiro.au/research/pulsar/psrcat/psrcat_help.html
    PARAMETERS: ClassVar[dict[str, ParameterDefinition]] = {
        # === Identification Parameters ===
        "JNAME": ParameterDefinition(
            code="JNAME",
            description="Pulsar name based on J2000 coordinates",
            unit=None,
            param_type=ParameterType.IDENTIFIER,
            category=ParameterCategory.IDENTIFICATION,
        ),
        "BNAME": ParameterDefinition(
            code="BNAME",
            description="Pulsar name based on B1950 coordinates",
            unit=None,
            param_type=ParameterType.IDENTIFIER,
            category=ParameterCategory.IDENTIFICATION,
        ),
        "NAME": ParameterDefinition(
            code="NAME",
            description="Common name (JNAME if exists, else BNAME)",
            unit=None,
            param_type=ParameterType.IDENTIFIER,
            category=ParameterCategory.IDENTIFICATION,
        ),
        # === Timing Parameters ===
        "P0": ParameterDefinition(
            code="P0",
            description="Barycentric period of the pulsar",
            unit="s",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            typical_range=(0.001, 12.0),
            related_params=("F0", "P1"),
            notes="Primary timing parameter. MSPs have P0 < 0.030 s",
        ),
        "P1": ParameterDefinition(
            code="P1",
            description="Time derivative of barycentic period",
            unit="",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            typical_range=(1e-22, 1e-10),
            related_params=("P0", "F1"),
            notes="Dimensionless. Also known as Pdot",
        ),
        "F0": ParameterDefinition(
            code="F0",
            description="Barycentric rotation frequency",
            unit="Hz",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            typical_range=(0.08, 716.0),
            related_params=("P0", "F1"),
        ),
        "F1": ParameterDefinition(
            code="F1",
            description="Time derivative of barycentric rotation frequency",
            unit="s^-2",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            typical_range=(-1e-9, -1e-17),
            related_params=("F0", "P1"),
            notes="Also known as Fdot. Usually negative (spin-down)",
        ),
        "F2": ParameterDefinition(
            code="F2",
            description="Second time derivative of barycentric rotation frequency",
            unit="s^-3",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            related_params=("F0", "F1"),
        ),
        "F3": ParameterDefinition(
            code="F3",
            description="Third time derivative of barycentric rotation frequency",
            unit="s^-4",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
        ),
        "PEPOCH": ParameterDefinition(
            code="PEPOCH",
            description="Epoch of period or frequency determination (MJD)",
            unit="d",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
        ),
        "DM": ParameterDefinition(
            code="DM",
            description="Dispersion measure",
            unit="pc cm^-3",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            typical_range=(0.0, 2000.0),
            notes="Integrated electron density along line of sight",
        ),
        "DM1": ParameterDefinition(
            code="DM1",
            description="First time derivative of dispersion measure",
            unit="pc cm^-3 yr^-1",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
        ),
        "RM": ParameterDefinition(
            code="RM",
            description="Rotation measure",
            unit="rad m^-2",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            notes="Measures magnetic field along line of sight",
        ),
        # === Astrometric Parameters ===
        "RAJD": ParameterDefinition(
            code="RAJD",
            description="Right ascension (J2000)",
            unit="deg",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
            typical_range=(0.0, 360.0),
        ),
        "DECJD": ParameterDefinition(
            code="DECJD",
            description="Declination (J2000)",
            unit="deg",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
            typical_range=(-90.0, 90.0),
        ),
        "RAJ": ParameterDefinition(
            code="RAJ",
            description="Right ascension (J2000) in HMS format",
            unit=None,
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "DECJ": ParameterDefinition(
            code="DECJ",
            description="Declination (J2000) in DMS format",
            unit=None,
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "PMRA": ParameterDefinition(
            code="PMRA",
            description="Proper motion in right ascension",
            unit="mas yr^-1",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "PMDEC": ParameterDefinition(
            code="PMDEC",
            description="Proper motion in declination",
            unit="mas yr^-1",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "PX": ParameterDefinition(
            code="PX",
            description="Annual parallax",
            unit="mas",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "POSEPOCH": ParameterDefinition(
            code="POSEPOCH",
            description="Epoch of position determination (MJD)",
            unit="d",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "ELONG": ParameterDefinition(
            code="ELONG",
            description="Ecliptic longitude",
            unit="deg",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "ELAT": ParameterDefinition(
            code="ELAT",
            description="Ecliptic latitude",
            unit="deg",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "GL": ParameterDefinition(
            code="GL",
            description="Galactic longitude",
            unit="deg",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
            typical_range=(0.0, 360.0),
        ),
        "GB": ParameterDefinition(
            code="GB",
            description="Galactic latitude",
            unit="deg",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
            typical_range=(-90.0, 90.0),
        ),
        "DIST": ParameterDefinition(
            code="DIST",
            description="Best estimate of pulsar distance",
            unit="kpc",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
            typical_range=(0.01, 50.0),
            notes="Based on parallax, DM model, or association",
        ),
        "DIST_DM": ParameterDefinition(
            code="DIST_DM",
            description="Distance based on DM and electron density model",
            unit="kpc",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "DMSINB": ParameterDefinition(
            code="DMSINB",
            description="DM times sin of Galactic latitude",
            unit="pc cm^-3",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "ZZ": ParameterDefinition(
            code="ZZ",
            description="Distance from Galactic plane",
            unit="kpc",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "XX": ParameterDefinition(
            code="XX",
            description="X-coordinate in Galactic frame",
            unit="kpc",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
        ),
        "YY": ParameterDefinition(
            code="YY",
            description="Y-coordinate in Galactic frame",
            unit="kpc",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.ASTROMETRIC,
        ),
        # === Binary Parameters ===
        "BINARY": ParameterDefinition(
            code="BINARY",
            description="Binary model used (e.g., BT, ELL1, DD)",
            unit=None,
            param_type=ParameterType.CATEGORICAL,
            category=ParameterCategory.BINARY,
            notes="Present only for binary pulsars",
        ),
        "PB": ParameterDefinition(
            code="PB",
            description="Orbital period",
            unit="d",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            typical_range=(0.06, 1000.0),
            notes="Binary pulsars only",
        ),
        "A1": ParameterDefinition(
            code="A1",
            description="Projected semi-major axis of pulsar orbit",
            unit="ls",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            notes="In light-seconds. Binary pulsars only",
        ),
        "E": ParameterDefinition(
            code="E",
            description="Orbital eccentricity (alternative name)",
            unit="",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            typical_range=(0.0, 1.0),
        ),
        "ECC": ParameterDefinition(
            code="ECC",
            description="Orbital eccentricity",
            unit="",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            typical_range=(0.0, 1.0),
            notes="Dimensionless. Most MSP binaries have ECC < 0.001",
        ),
        "T0": ParameterDefinition(
            code="T0",
            description="Epoch of periastron (MJD)",
            unit="d",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
        ),
        "TASC": ParameterDefinition(
            code="TASC",
            description="Epoch of ascending node (MJD)",
            unit="d",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            notes="Used in ELL1 binary model",
        ),
        "OM": ParameterDefinition(
            code="OM",
            description="Longitude of periastron",
            unit="deg",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
        ),
        "OMDOT": ParameterDefinition(
            code="OMDOT",
            description="Rate of periastron advance",
            unit="deg yr^-1",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            notes="Relativistic effect, measurable in compact binaries",
        ),
        "PBDOT": ParameterDefinition(
            code="PBDOT",
            description="Time derivative of orbital period",
            unit="",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            notes="Dimensionless. Gravitational wave emission causes negative PBDOT",
        ),
        "GAMMA": ParameterDefinition(
            code="GAMMA",
            description="Post-Keplerian gravitational redshift parameter",
            unit="s",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
        ),
        "SINI": ParameterDefinition(
            code="SINI",
            description="Sine of orbital inclination",
            unit="",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
            typical_range=(0.0, 1.0),
        ),
        "M2": ParameterDefinition(
            code="M2",
            description="Companion mass (from Shapiro delay)",
            unit="M_sun",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.BINARY,
        ),
        "MASSFN": ParameterDefinition(
            code="MASSFN",
            description="Pulsar mass function",
            unit="M_sun",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.BINARY,
            formula="(4*pi^2/G) * (A1*c)^3 / PB^2",
        ),
        "MINMASS": ParameterDefinition(
            code="MINMASS",
            description="Minimum companion mass (assuming i=90, Mp=1.4)",
            unit="M_sun",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.BINARY,
        ),
        "MEDMASS": ParameterDefinition(
            code="MEDMASS",
            description="Median companion mass (assuming i=60, Mp=1.4)",
            unit="M_sun",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.BINARY,
        ),
        "BINCOMP": ParameterDefinition(
            code="BINCOMP",
            description="Binary companion type",
            unit=None,
            param_type=ParameterType.CATEGORICAL,
            category=ParameterCategory.BINARY,
            notes="e.g., MS (main sequence), WD (white dwarf), NS (neutron star)",
        ),
        # === Derived Parameters ===
        "BSURF": ParameterDefinition(
            code="BSURF",
            description="Surface magnetic field strength",
            unit="G",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            typical_range=(1e7, 1e15),
            formula="3.2e19 * sqrt(P0 * P1)",
            related_params=("P0", "P1"),
            notes="Assumes magnetic dipole braking",
        ),
        "B_LC": ParameterDefinition(
            code="B_LC",
            description="Magnetic field at light cylinder",
            unit="G",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            related_params=("P0", "BSURF"),
        ),
        "EDOT": ParameterDefinition(
            code="EDOT",
            description="Spin-down energy loss rate",
            unit="erg s^-1",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            typical_range=(1e26, 5e38),
            formula="4*pi^2 * I * P1 / P0^3",
            related_params=("P0", "P1"),
            notes="Assumes I = 1e45 g cm^2",
        ),
        "EDOTD2": ParameterDefinition(
            code="EDOTD2",
            description="Energy flux at Sun (Edot / D^2)",
            unit="erg s^-1 kpc^-2",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            related_params=("EDOT", "DIST"),
        ),
        "AGE": ParameterDefinition(
            code="AGE",
            description="Characteristic age (spin-down age)",
            unit="yr",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            typical_range=(100.0, 1e11),
            formula="P0 / (2 * P1)",
            related_params=("P0", "P1"),
            notes="Assumes n=3 braking index and P0_initial << P0",
        ),
        "PMTOT": ParameterDefinition(
            code="PMTOT",
            description="Total proper motion",
            unit="mas yr^-1",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            formula="sqrt(PMRA^2 + PMDEC^2)",
            related_params=("PMRA", "PMDEC"),
        ),
        "VTRANS": ParameterDefinition(
            code="VTRANS",
            description="Transverse velocity",
            unit="km s^-1",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            related_params=("PMTOT", "DIST"),
        ),
        "P1_I": ParameterDefinition(
            code="P1_I",
            description="Intrinsic period derivative (Shklovskii corrected)",
            unit="",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
            notes="Corrected for kinematic effects",
        ),
        "AGE_I": ParameterDefinition(
            code="AGE_I",
            description="Intrinsic age (Shklovskii corrected)",
            unit="yr",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
        ),
        "BSURF_I": ParameterDefinition(
            code="BSURF_I",
            description="Intrinsic surface magnetic field (Shklovskii corrected)",
            unit="G",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.DERIVED,
        ),
        # === Association Parameters ===
        "ASSOC": ParameterDefinition(
            code="ASSOC",
            description="Associations with SNRs, globular clusters, etc.",
            unit=None,
            param_type=ParameterType.CATEGORICAL,
            category=ParameterCategory.ASSOCIATION,
            notes="Format: TYPE:NAME (e.g., GC:47Tuc, SNR:Vela)",
        ),
        "TYPE": ParameterDefinition(
            code="TYPE",
            description="Pulsar type classification",
            unit=None,
            param_type=ParameterType.CATEGORICAL,
            category=ParameterCategory.ASSOCIATION,
            notes="e.g., RRAT, AXP, XINS, HE, NRAD",
        ),
        # === Survey/Observation Parameters ===
        "S400": ParameterDefinition(
            code="S400",
            description="Mean flux density at 400 MHz",
            unit="mJy",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.SURVEY,
        ),
        "S1400": ParameterDefinition(
            code="S1400",
            description="Mean flux density at 1400 MHz",
            unit="mJy",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.SURVEY,
        ),
        "S2000": ParameterDefinition(
            code="S2000",
            description="Mean flux density at 2000 MHz",
            unit="mJy",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.SURVEY,
        ),
        "W50": ParameterDefinition(
            code="W50",
            description="Width of pulse at 50% of peak",
            unit="ms",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.SURVEY,
        ),
        "W10": ParameterDefinition(
            code="W10",
            description="Width of pulse at 10% of peak",
            unit="ms",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.SURVEY,
        ),
        "SPINDX": ParameterDefinition(
            code="SPINDX",
            description="Spectral index",
            unit="",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.SURVEY,
            notes="S ~ nu^SPINDX",
        ),
        "SURVEY": ParameterDefinition(
            code="SURVEY",
            description="Surveys that detected this pulsar",
            unit=None,
            param_type=ParameterType.CATEGORICAL,
            category=ParameterCategory.SURVEY,
        ),
        "NGLT": ParameterDefinition(
            code="NGLT",
            description="Number of glitches observed",
            unit="",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
        ),
        "R_LUM": ParameterDefinition(
            code="R_LUM",
            description="Radio luminosity at 400 MHz",
            unit="mJy kpc^2",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.SURVEY,
        ),
        "R_LUM14": ParameterDefinition(
            code="R_LUM14",
            description="Radio luminosity at 1400 MHz",
            unit="mJy kpc^2",
            param_type=ParameterType.DERIVED_ATNF,
            category=ParameterCategory.SURVEY,
        ),
    }

    # Human-readable aliases mapping to canonical parameter codes
    # These are used for natural language processing
    ALIASES: ClassVar[dict[str, list[str]]] = {
        # Identifiers
        "JNAME": ["name", "pulsar name", "J name", "J2000 name"],
        "BNAME": ["B name", "B1950 name", "old name"],
        # Timing
        "P0": [
            "period",
            "spin period",
            "rotation period",
            "pulse period",
            "P",
            "rotational period",
        ],
        "P1": [
            "period derivative",
            "Pdot",
            "P-dot",
            "spin-down rate",
            "P dot",
            "first period derivative",
        ],
        "F0": [
            "frequency",
            "spin frequency",
            "rotation frequency",
            "rotational frequency",
            "nu",
        ],
        "F1": [
            "frequency derivative",
            "Fdot",
            "F-dot",
            "F dot",
            "first frequency derivative",
        ],
        "DM": [
            "dispersion measure",
            "dispersion",
            "DM",
        ],
        "RM": [
            "rotation measure",
            "Faraday rotation",
        ],
        # Astrometric
        "RAJD": ["right ascension", "RA", "R.A.", "alpha"],
        "DECJD": ["declination", "Dec", "DEC", "delta"],
        "GL": ["galactic longitude", "l", "gal lon"],
        "GB": ["galactic latitude", "b", "gal lat"],
        "DIST": ["distance", "D", "dist"],
        "PMRA": ["proper motion RA", "PM RA", "proper motion in RA"],
        "PMDEC": ["proper motion Dec", "PM Dec", "proper motion in declination"],
        "PX": ["parallax", "annual parallax"],
        # Binary
        "BINARY": ["binary model", "binary type", "orbit model"],
        "PB": ["orbital period", "binary period", "Porb", "P_orb"],
        "A1": ["semi-major axis", "projected semi-major axis", "a sin i", "asini"],
        "ECC": ["eccentricity", "orbital eccentricity", "e"],
        "T0": ["epoch of periastron", "periastron epoch"],
        "OM": ["longitude of periastron", "omega", "argument of periastron"],
        "OMDOT": ["periastron advance", "omega dot", "apsidal motion"],
        "PBDOT": ["orbital period derivative", "Pb dot", "orbital decay"],
        "SINI": ["sin i", "sine of inclination", "orbital inclination"],
        "M2": ["companion mass", "secondary mass"],
        "BINCOMP": ["companion type", "binary companion", "companion"],
        "MASSFN": ["mass function"],
        # Derived
        "BSURF": [
            "magnetic field",
            "surface magnetic field",
            "B field",
            "B-field",
            "surface field",
            "B",
        ],
        "B_LC": ["light cylinder field", "B_LC"],
        "EDOT": [
            "spin-down luminosity",
            "energy loss rate",
            "Edot",
            "E-dot",
            "E dot",
            "spindown luminosity",
            "spin-down power",
        ],
        "AGE": [
            "characteristic age",
            "spin-down age",
            "tau",
            "tau_c",
            "age",
        ],
        "PMTOT": ["total proper motion", "proper motion"],
        "VTRANS": ["transverse velocity", "space velocity", "velocity"],
        # Association
        "ASSOC": [
            "association",
            "associations",
            "cluster",
            "SNR",
            "supernova remnant",
            "globular cluster",
        ],
        "TYPE": ["pulsar type", "classification", "type"],
        # Survey
        "S400": ["flux 400", "400 MHz flux", "S_400"],
        "S1400": ["flux 1400", "1.4 GHz flux", "S_1400", "flux"],
        "W50": ["pulse width", "W_50", "width at 50%"],
        "W10": ["W_10", "width at 10%"],
        "SPINDX": ["spectral index", "alpha", "spectrum"],
    }

    # Astropy unit registry
    UNIT_REGISTRY: ClassVar[dict[str, u.Unit | None]] = {
        # Timing
        "P0": u.s,
        "P1": u.dimensionless_unscaled,
        "F0": u.Hz,
        "F1": u.Hz / u.s,
        "F2": u.Hz / u.s**2,
        "DM": u.pc / u.cm**3,
        "RM": u.rad / u.m**2,
        "PEPOCH": u.d,
        # Astrometric
        "RAJD": u.deg,
        "DECJD": u.deg,
        "GL": u.deg,
        "GB": u.deg,
        "PMRA": u.mas / u.yr,
        "PMDEC": u.mas / u.yr,
        "PX": u.mas,
        "DIST": u.kpc,
        "POSEPOCH": u.d,
        # Binary
        "PB": u.d,
        "A1": u.lyr / (365.25 * 86400) * u.s,  # light-seconds
        "ECC": u.dimensionless_unscaled,
        "T0": u.d,
        "OM": u.deg,
        "OMDOT": u.deg / u.yr,
        "PBDOT": u.dimensionless_unscaled,
        "SINI": u.dimensionless_unscaled,
        "M2": u.Msun,
        "MASSFN": u.Msun,
        # Derived
        "BSURF": u.G,
        "EDOT": u.erg / u.s,
        "AGE": u.yr,
        "PMTOT": u.mas / u.yr,
        "VTRANS": u.km / u.s,
        # Survey
        "S400": u.mJy,
        "S1400": u.mJy,
        "W50": u.ms,
        "W10": u.ms,
        "SPINDX": u.dimensionless_unscaled,
    }

    def __init__(self) -> None:
        """Initialize the schema grounding pack."""
        # Build reverse alias lookup
        self._alias_to_code: dict[str, str] = {}
        for code, aliases in self.ALIASES.items():
            for alias in aliases:
                self._alias_to_code[alias.lower()] = code
            # Also add the code itself as an alias (case-insensitive)
            self._alias_to_code[code.lower()] = code

    def get_parameter(self, code: str) -> ParameterDefinition | None:
        """Get parameter definition by canonical code.

        Args:
            code: The ATNF parameter code (case-insensitive)

        Returns:
            ParameterDefinition if found, None otherwise
        """
        return self.PARAMETERS.get(code.upper())

    def resolve_alias(self, alias: str) -> str | None:
        """Resolve a human-readable alias to canonical parameter code.

        Args:
            alias: Human-readable parameter name or alias

        Returns:
            Canonical ATNF parameter code if found, None otherwise

        Example:
            >>> schema = SchemaGroundingPack()
            >>> schema.resolve_alias("spin period")
            'P0'
            >>> schema.resolve_alias("magnetic field")
            'BSURF'
        """
        return self._alias_to_code.get(alias.lower())

    def get_aliases(self, code: str) -> list[str]:
        """Get human-readable aliases for a parameter code.

        Args:
            code: ATNF parameter code

        Returns:
            List of human-readable aliases for the parameter

        Example:
            >>> schema = SchemaGroundingPack()
            >>> schema.get_aliases("P0")
            ['period', 'spin period', 'rotation period', ...]
        """
        return self.ALIASES.get(code.upper(), [])

    def is_valid_parameter(self, code: str) -> bool:
        """Check if a parameter code is valid.

        Args:
            code: Parameter code to validate

        Returns:
            True if valid ATNF parameter code
        """
        return code.upper() in self.PARAMETERS

    def get_unit(self, code: str) -> u.Unit | None:
        """Get astropy unit for a parameter.

        Args:
            code: ATNF parameter code

        Returns:
            astropy Unit object or None if dimensionless/categorical
        """
        return self.UNIT_REGISTRY.get(code.upper())

    def get_all_codes(self) -> list[str]:
        """Get list of all valid parameter codes.

        Returns:
            List of canonical ATNF parameter codes
        """
        return list(self.PARAMETERS.keys())

    def get_parameters_by_category(
        self, category: ParameterCategory
    ) -> list[ParameterDefinition]:
        """Get all parameters in a category.

        Args:
            category: Parameter category to filter by

        Returns:
            List of ParameterDefinition objects in that category
        """
        return [p for p in self.PARAMETERS.values() if p.category == category]

    def get_parameters_by_type(
        self, param_type: ParameterType
    ) -> list[ParameterDefinition]:
        """Get all parameters of a specific type.

        Args:
            param_type: Parameter type to filter by

        Returns:
            List of ParameterDefinition objects of that type
        """
        return [p for p in self.PARAMETERS.values() if p.param_type == param_type]

    def get_derived_atnf_parameters(self) -> list[str]:
        """Get codes for parameters derived by ATNF (available in catalogue).

        Returns:
            List of parameter codes that ATNF computes and includes
        """
        return [
            p.code
            for p in self.PARAMETERS.values()
            if p.param_type == ParameterType.DERIVED_ATNF
        ]

    def validate_value_range(self, code: str, value: float) -> tuple[bool, str | None]:
        """Check if a value is within typical range for a parameter.

        Args:
            code: Parameter code
            value: Value to validate

        Returns:
            Tuple of (is_valid, warning_message)
            Warning is provided if value is outside typical range but not rejected
        """
        param = self.get_parameter(code)
        if param is None:
            return False, f"Unknown parameter: {code}"

        if param.typical_range is None:
            return True, None

        min_val, max_val = param.typical_range
        if value < min_val or value > max_val:
            return True, (
                f"Value {value} for {code} is outside typical range "
                f"[{min_val}, {max_val}]. This may indicate an unusual pulsar "
                f"or a unit conversion issue."
            )

        return True, None

    def get_llm_context(self) -> str:
        """Generate LLM system prompt context with parameter information.

        Returns:
            Formatted string for LLM system prompt
        """
        lines = [
            "## ATNF Pulsar Catalogue Parameters",
            "",
            "### Canonical Parameter Codes (use these in queries):",
            "",
        ]

        # Group by category
        for category in ParameterCategory:
            params = self.get_parameters_by_category(category)
            if not params:
                continue

            lines.append(f"**{category.value.title()}:**")
            for p in params:
                unit_str = f" ({p.unit})" if p.unit else ""
                lines.append(f"- `{p.code}`: {p.description}{unit_str}")
            lines.append("")

        # Add common aliases
        lines.extend(
            [
                "### Common Natural Language Aliases:",
                "",
                "| Alias | Parameter Code |",
                "|-------|----------------|",
            ]
        )
        common_aliases = [
            ("period", "P0"),
            ("Pdot", "P1"),
            ("frequency", "F0"),
            ("dispersion measure", "DM"),
            ("magnetic field", "BSURF"),
            ("spin-down luminosity", "EDOT"),
            ("characteristic age", "AGE"),
            ("orbital period", "PB"),
            ("eccentricity", "ECC"),
            ("distance", "DIST"),
        ]
        for alias, code in common_aliases:
            lines.append(f"| {alias} | {code} |")

        return "\n".join(lines)

    def export_mappings(self, filepath: str) -> None:
        """Export parameter mappings to JSON for LLM system prompt.

        Args:
            filepath: Path to write JSON file
        """
        import json

        mappings = {}
        for code, param in self.PARAMETERS.items():
            mappings[code] = {
                "canonical": code,
                "description": param.description,
                "unit": param.unit,
                "type": param.param_type.value,
                "category": param.category.value,
                "aliases": self.ALIASES.get(code, []),
            }
            if param.typical_range:
                mappings[code]["typical_range"] = list(param.typical_range)
            if param.formula:
                mappings[code]["formula"] = param.formula

        with open(filepath, "w") as f:
            json.dump(mappings, f, indent=2)
