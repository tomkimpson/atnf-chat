"""Tests for the Schema Grounding Pack module."""

import pytest
from astropy import units as u

from atnf_chat.core.schema import (
    ParameterCategory,
    ParameterDefinition,
    ParameterType,
    SchemaGroundingPack,
)


class TestSchemaGroundingPack:
    """Test suite for SchemaGroundingPack class."""

    @pytest.fixture
    def schema(self) -> SchemaGroundingPack:
        """Create a SchemaGroundingPack instance for testing."""
        return SchemaGroundingPack()

    # === Parameter Definition Tests ===

    def test_all_parameters_have_required_fields(self, schema: SchemaGroundingPack) -> None:
        """Verify all parameter definitions have required fields."""
        for code, param in schema.PARAMETERS.items():
            assert isinstance(param, ParameterDefinition)
            assert param.code == code
            assert param.description
            assert isinstance(param.param_type, ParameterType)
            assert isinstance(param.category, ParameterCategory)

    def test_core_parameters_exist(self, schema: SchemaGroundingPack) -> None:
        """Verify core pulsar parameters are defined."""
        core_params = [
            "JNAME",
            "P0",
            "P1",
            "F0",
            "F1",
            "DM",
            "RAJD",
            "DECJD",
            "BSURF",
            "EDOT",
            "AGE",
            "PB",
            "ECC",
            "ASSOC",
        ]
        for param in core_params:
            assert param in schema.PARAMETERS, f"Missing core parameter: {param}"

    def test_derived_atnf_parameters_have_formula(
        self, schema: SchemaGroundingPack
    ) -> None:
        """Verify key derived parameters document their formula."""
        params_with_formulas = ["BSURF", "EDOT", "AGE", "MASSFN"]
        for code in params_with_formulas:
            param = schema.get_parameter(code)
            assert param is not None
            assert param.param_type == ParameterType.DERIVED_ATNF
            # Some derived params should have formula documented
            if code in ["BSURF", "EDOT", "AGE"]:
                assert param.formula is not None, f"{code} should have formula"

    # === Alias Resolution Tests ===

    def test_resolve_common_aliases(self, schema: SchemaGroundingPack) -> None:
        """Test resolution of common natural language aliases."""
        alias_tests = [
            ("period", "P0"),
            ("spin period", "P0"),
            ("Pdot", "P1"),
            ("period derivative", "P1"),
            ("frequency", "F0"),
            ("dispersion measure", "DM"),
            ("magnetic field", "BSURF"),
            ("spin-down luminosity", "EDOT"),
            ("characteristic age", "AGE"),
            ("orbital period", "PB"),
            ("eccentricity", "ECC"),
            ("distance", "DIST"),
        ]
        for alias, expected_code in alias_tests:
            result = schema.resolve_alias(alias)
            assert result == expected_code, f"Alias '{alias}' should resolve to {expected_code}"

    def test_resolve_alias_case_insensitive(self, schema: SchemaGroundingPack) -> None:
        """Test that alias resolution is case-insensitive."""
        assert schema.resolve_alias("Period") == "P0"
        assert schema.resolve_alias("PERIOD") == "P0"
        assert schema.resolve_alias("Spin Period") == "P0"
        assert schema.resolve_alias("DM") == "DM"
        assert schema.resolve_alias("dm") == "DM"

    def test_resolve_alias_canonical_codes(self, schema: SchemaGroundingPack) -> None:
        """Test that canonical codes resolve to themselves."""
        for code in ["P0", "F0", "DM", "BSURF", "EDOT"]:
            assert schema.resolve_alias(code) == code
            assert schema.resolve_alias(code.lower()) == code

    def test_resolve_alias_unknown_returns_none(
        self, schema: SchemaGroundingPack
    ) -> None:
        """Test that unknown aliases return None."""
        assert schema.resolve_alias("unknown_parameter") is None
        assert schema.resolve_alias("not_a_real_thing") is None

    # === Parameter Validation Tests ===

    def test_is_valid_parameter(self, schema: SchemaGroundingPack) -> None:
        """Test parameter validation."""
        assert schema.is_valid_parameter("P0")
        assert schema.is_valid_parameter("p0")  # Case insensitive
        assert schema.is_valid_parameter("BSURF")
        assert not schema.is_valid_parameter("INVALID")
        assert not schema.is_valid_parameter("")

    def test_get_parameter_returns_definition(
        self, schema: SchemaGroundingPack
    ) -> None:
        """Test getting parameter definitions."""
        p0 = schema.get_parameter("P0")
        assert p0 is not None
        assert p0.code == "P0"
        assert p0.description == "Barycentric period of the pulsar"
        assert p0.unit == "s"
        assert p0.param_type == ParameterType.MEASURED
        assert p0.category == ParameterCategory.TIMING

    def test_get_parameter_case_insensitive(
        self, schema: SchemaGroundingPack
    ) -> None:
        """Test that get_parameter is case-insensitive."""
        assert schema.get_parameter("p0") == schema.get_parameter("P0")
        assert schema.get_parameter("bsurf") == schema.get_parameter("BSURF")

    def test_get_parameter_unknown_returns_none(
        self, schema: SchemaGroundingPack
    ) -> None:
        """Test that unknown parameters return None."""
        assert schema.get_parameter("INVALID") is None
        assert schema.get_parameter("") is None

    # === Unit Registry Tests ===

    def test_get_unit_timing_parameters(self, schema: SchemaGroundingPack) -> None:
        """Test units for timing parameters."""
        assert schema.get_unit("P0") == u.s
        assert schema.get_unit("F0") == u.Hz
        assert schema.get_unit("DM") == u.pc / u.cm**3

    def test_get_unit_derived_parameters(self, schema: SchemaGroundingPack) -> None:
        """Test units for derived parameters."""
        assert schema.get_unit("BSURF") == u.G
        assert schema.get_unit("EDOT") == u.erg / u.s
        assert schema.get_unit("AGE") == u.yr

    def test_get_unit_dimensionless(self, schema: SchemaGroundingPack) -> None:
        """Test dimensionless parameters."""
        assert schema.get_unit("P1") == u.dimensionless_unscaled
        assert schema.get_unit("ECC") == u.dimensionless_unscaled

    def test_get_unit_unknown_returns_none(self, schema: SchemaGroundingPack) -> None:
        """Test that unknown parameters return None for unit."""
        assert schema.get_unit("INVALID") is None
        assert schema.get_unit("JNAME") is None  # Identifier, no unit in registry

    # === Value Range Validation Tests ===

    def test_validate_value_range_normal(self, schema: SchemaGroundingPack) -> None:
        """Test validation of values within normal range."""
        # Normal pulsar period
        is_valid, warning = schema.validate_value_range("P0", 0.5)
        assert is_valid
        assert warning is None

        # MSP period
        is_valid, warning = schema.validate_value_range("P0", 0.005)
        assert is_valid
        assert warning is None

    def test_validate_value_range_unusual(self, schema: SchemaGroundingPack) -> None:
        """Test validation flags unusual values."""
        # Very long period (outside typical range)
        is_valid, warning = schema.validate_value_range("P0", 100.0)
        assert is_valid  # Still valid, but with warning
        assert warning is not None
        assert "outside typical range" in warning

    def test_validate_value_range_unknown_param(
        self, schema: SchemaGroundingPack
    ) -> None:
        """Test validation fails for unknown parameters."""
        is_valid, warning = schema.validate_value_range("INVALID", 1.0)
        assert not is_valid
        assert "Unknown parameter" in warning

    # === Category and Type Filtering Tests ===

    def test_get_parameters_by_category(self, schema: SchemaGroundingPack) -> None:
        """Test filtering parameters by category."""
        timing_params = schema.get_parameters_by_category(ParameterCategory.TIMING)
        assert len(timing_params) > 0
        assert all(p.category == ParameterCategory.TIMING for p in timing_params)

        # Verify specific params are in expected categories
        timing_codes = [p.code for p in timing_params]
        assert "P0" in timing_codes
        assert "F0" in timing_codes
        assert "DM" in timing_codes

    def test_get_parameters_by_type(self, schema: SchemaGroundingPack) -> None:
        """Test filtering parameters by type."""
        measured = schema.get_parameters_by_type(ParameterType.MEASURED)
        assert len(measured) > 0
        assert all(p.param_type == ParameterType.MEASURED for p in measured)

        derived = schema.get_parameters_by_type(ParameterType.DERIVED_ATNF)
        assert len(derived) > 0
        assert all(p.param_type == ParameterType.DERIVED_ATNF for p in derived)

    def test_get_derived_atnf_parameters(self, schema: SchemaGroundingPack) -> None:
        """Test getting ATNF-derived parameter codes."""
        derived_codes = schema.get_derived_atnf_parameters()
        assert "BSURF" in derived_codes
        assert "EDOT" in derived_codes
        assert "AGE" in derived_codes
        assert "DIST" in derived_codes

        # Measured params should not be in this list
        assert "P0" not in derived_codes
        assert "DM" not in derived_codes

    # === Utility Method Tests ===

    def test_get_all_codes(self, schema: SchemaGroundingPack) -> None:
        """Test getting all parameter codes."""
        codes = schema.get_all_codes()
        assert len(codes) > 50  # Should have many parameters
        assert "P0" in codes
        assert "BSURF" in codes
        assert "JNAME" in codes

    def test_get_llm_context(self, schema: SchemaGroundingPack) -> None:
        """Test LLM context generation."""
        context = schema.get_llm_context()
        assert isinstance(context, str)
        assert "ATNF Pulsar Catalogue Parameters" in context
        assert "P0" in context
        assert "BSURF" in context
        assert "period" in context.lower()

    def test_export_mappings(self, schema: SchemaGroundingPack, tmp_path) -> None:
        """Test exporting mappings to JSON."""
        import json

        filepath = tmp_path / "mappings.json"
        schema.export_mappings(str(filepath))

        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert "P0" in data
        assert data["P0"]["canonical"] == "P0"
        assert data["P0"]["unit"] == "s"
        assert "period" in data["P0"]["aliases"]


class TestParameterDefinition:
    """Tests for ParameterDefinition dataclass."""

    def test_parameter_definition_creation(self) -> None:
        """Test creating a ParameterDefinition."""
        param = ParameterDefinition(
            code="TEST",
            description="Test parameter",
            unit="s",
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
            typical_range=(0.001, 10.0),
        )
        assert param.code == "TEST"
        assert param.description == "Test parameter"
        assert param.unit == "s"
        assert param.typical_range == (0.001, 10.0)

    def test_parameter_definition_immutable(self) -> None:
        """Test that ParameterDefinition is immutable (frozen)."""
        param = ParameterDefinition(
            code="TEST",
            description="Test",
            unit=None,
            param_type=ParameterType.MEASURED,
            category=ParameterCategory.TIMING,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            param.code = "CHANGED"


class TestParameterEnums:
    """Tests for parameter type and category enums."""

    def test_parameter_type_values(self) -> None:
        """Test ParameterType enum values."""
        assert ParameterType.MEASURED.value == "measured"
        assert ParameterType.DERIVED_ATNF.value == "derived_atnf"
        assert ParameterType.DERIVED_CUSTOM.value == "derived_custom"
        assert ParameterType.CATEGORICAL.value == "categorical"
        assert ParameterType.IDENTIFIER.value == "identifier"

    def test_parameter_category_values(self) -> None:
        """Test ParameterCategory enum values."""
        assert ParameterCategory.TIMING.value == "timing"
        assert ParameterCategory.ASTROMETRIC.value == "astrometric"
        assert ParameterCategory.BINARY.value == "binary"
        assert ParameterCategory.DERIVED.value == "derived"
