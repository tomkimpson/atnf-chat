"""Tests for visualization plots module."""

import numpy as np
import pandas as pd
import pytest

from atnf_chat.visualization.plots import (
    PlotResult,
    create_comparison_plot,
    create_histogram,
    create_pp_diagram,
    create_scatter_plot,
    create_sky_plot,
)


@pytest.fixture
def pulsar_df() -> pd.DataFrame:
    """Create sample pulsar DataFrame for testing."""
    np.random.seed(42)
    n = 50

    return pd.DataFrame({
        "JNAME": [f"J{i:04d}+0000" for i in range(n)],
        "P0": np.random.lognormal(-2, 1, n),  # Period in seconds
        "P1": np.random.lognormal(-35, 2, n),  # Period derivative
        "F0": 1 / np.random.lognormal(-2, 1, n),  # Frequency
        "DM": np.random.exponential(50, n),
        "DIST": np.random.exponential(2, n),
        "RAJD": np.random.uniform(0, 360, n),
        "DECJD": np.random.uniform(-90, 90, n),
        "GL": np.random.uniform(0, 360, n),
        "GB": np.random.uniform(-90, 90, n),
        "BSURF": np.random.lognormal(28, 2, n),
        "TYPE": np.random.choice(["RADIO", "MSP", "BINARY"], n),
    })


class TestPlotResult:
    """Tests for PlotResult dataclass."""

    def test_create_result(self, pulsar_df: pd.DataFrame) -> None:
        """Test that PlotResult has expected attributes."""
        result = create_pp_diagram(pulsar_df)
        assert hasattr(result, 'figure')
        assert hasattr(result, 'title')
        assert hasattr(result, 'description')
        assert hasattr(result, 'data_summary')

    def test_to_html(self, pulsar_df: pd.DataFrame) -> None:
        """Test converting to HTML."""
        result = create_histogram(pulsar_df, parameter="P0")
        html = result.to_html()
        assert isinstance(html, str)
        assert len(html) > 0

    def test_to_json(self, pulsar_df: pd.DataFrame) -> None:
        """Test converting to JSON."""
        result = create_histogram(pulsar_df, parameter="P0")
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0


class TestPPDiagram:
    """Tests for P-Pdot diagram creation."""

    def test_create_pp_diagram_basic(self, pulsar_df: pd.DataFrame) -> None:
        """Test creating a basic P-Pdot diagram."""
        result = create_pp_diagram(pulsar_df)

        assert result.figure is not None
        assert "Period" in result.title or "P-Pdot" in result.title
        assert result.data_summary["total_pulsars"] == len(pulsar_df)

    def test_pp_diagram_with_highlight_groups(self, pulsar_df: pd.DataFrame) -> None:
        """Test P-Pdot diagram with highlighted groups."""
        result = create_pp_diagram(
            pulsar_df,
            highlight_groups={"MSPs": "P0 < 0.1", "Slow": "P0 >= 0.1"},
        )

        assert result.figure is not None

    def test_pp_diagram_without_lines(self, pulsar_df: pd.DataFrame) -> None:
        """Test P-Pdot diagram without constant B and age lines."""
        result = create_pp_diagram(pulsar_df, show_lines=False)

        assert result.figure is not None

    def test_pp_diagram_missing_columns(self) -> None:
        """Test P-Pdot diagram with missing required columns."""
        df = pd.DataFrame({"JNAME": ["J0001"]})

        with pytest.raises(ValueError, match="P0 and P1"):
            create_pp_diagram(df)

    def test_pp_diagram_custom_title(self, pulsar_df: pd.DataFrame) -> None:
        """Test P-Pdot diagram with custom title."""
        result = create_pp_diagram(pulsar_df, title="Custom Title")
        assert result.title == "Custom Title"

    def test_pp_diagram_data_summary(self, pulsar_df: pd.DataFrame) -> None:
        """Test P-Pdot diagram data summary."""
        result = create_pp_diagram(pulsar_df)

        assert "total_pulsars" in result.data_summary
        assert "plotted_pulsars" in result.data_summary
        assert "p0_range" in result.data_summary
        assert "p1_range" in result.data_summary


class TestHistogram:
    """Tests for histogram creation."""

    def test_create_histogram_basic(self, pulsar_df: pd.DataFrame) -> None:
        """Test creating a basic histogram."""
        result = create_histogram(pulsar_df, parameter="P0")

        assert result.figure is not None
        assert "P0" in result.title

    def test_histogram_log_scale(self, pulsar_df: pd.DataFrame) -> None:
        """Test histogram with log scale."""
        result = create_histogram(pulsar_df, parameter="P0", log_scale=True)

        assert result.figure is not None

    def test_histogram_custom_bins(self, pulsar_df: pd.DataFrame) -> None:
        """Test histogram with custom number of bins."""
        result = create_histogram(pulsar_df, parameter="DM", nbins=20)

        assert result.figure is not None

    def test_histogram_missing_parameter(self, pulsar_df: pd.DataFrame) -> None:
        """Test histogram with missing parameter."""
        with pytest.raises(ValueError, match="NONEXISTENT"):
            create_histogram(pulsar_df, parameter="NONEXISTENT")

    def test_histogram_data_summary(self, pulsar_df: pd.DataFrame) -> None:
        """Test histogram data summary."""
        result = create_histogram(pulsar_df, parameter="DM")

        assert "parameter" in result.data_summary
        assert "count" in result.data_summary
        assert "mean" in result.data_summary
        assert "median" in result.data_summary


class TestScatterPlot:
    """Tests for scatter plot creation."""

    def test_create_scatter_basic(self, pulsar_df: pd.DataFrame) -> None:
        """Test creating a basic scatter plot."""
        result = create_scatter_plot(pulsar_df, x_param="P0", y_param="DM")

        assert result.figure is not None

    def test_scatter_with_color(self, pulsar_df: pd.DataFrame) -> None:
        """Test scatter plot with color coding."""
        result = create_scatter_plot(
            pulsar_df, x_param="P0", y_param="DM", color_by="DIST"
        )

        assert result.figure is not None

    def test_scatter_log_scales(self, pulsar_df: pd.DataFrame) -> None:
        """Test scatter plot with log scales."""
        result = create_scatter_plot(
            pulsar_df, x_param="P0", y_param="P1", log_x=True, log_y=True
        )

        assert result.figure is not None

    def test_scatter_with_regression(self, pulsar_df: pd.DataFrame) -> None:
        """Test scatter plot with regression line."""
        result = create_scatter_plot(
            pulsar_df,
            x_param="P0",
            y_param="DM",
            show_regression=True,
        )

        assert result.figure is not None

    def test_scatter_missing_parameter(self, pulsar_df: pd.DataFrame) -> None:
        """Test scatter plot with missing parameter."""
        with pytest.raises(ValueError, match="NONEXISTENT"):
            create_scatter_plot(pulsar_df, x_param="NONEXISTENT", y_param="DM")

    def test_scatter_data_summary(self, pulsar_df: pd.DataFrame) -> None:
        """Test scatter plot data summary."""
        result = create_scatter_plot(pulsar_df, x_param="P0", y_param="DM")

        assert "x_param" in result.data_summary
        assert "y_param" in result.data_summary
        assert "n_points" in result.data_summary


class TestSkyPlot:
    """Tests for sky distribution plot creation."""

    def test_create_sky_plot_galactic(self, pulsar_df: pd.DataFrame) -> None:
        """Test creating a Galactic sky plot."""
        result = create_sky_plot(pulsar_df)

        assert result.figure is not None
        assert "Galactic" in result.data_summary["coordinate_system"]

    def test_sky_plot_with_color(self, pulsar_df: pd.DataFrame) -> None:
        """Test sky plot with color coding."""
        result = create_sky_plot(pulsar_df, color_by="DM")

        assert result.figure is not None

    def test_sky_plot_equirectangular(self, pulsar_df: pd.DataFrame) -> None:
        """Test sky plot with equirectangular projection."""
        result = create_sky_plot(pulsar_df, projection="equirectangular")

        assert result.figure is not None

    def test_sky_plot_with_equatorial_coords(self) -> None:
        """Test sky plot with equatorial coordinates when Galactic missing."""
        df = pd.DataFrame({
            "JNAME": ["J0001", "J0002"],
            "RAJD": [10.0, 20.0],
            "DECJD": [5.0, -5.0],
        })
        result = create_sky_plot(df)

        assert result.figure is not None
        assert "Equatorial" in result.data_summary["coordinate_system"]

    def test_sky_plot_missing_coordinates(self) -> None:
        """Test sky plot with missing coordinate columns."""
        df = pd.DataFrame({"JNAME": ["J0001"]})

        with pytest.raises(ValueError, match="GL/GB|RAJD/DECJD"):
            create_sky_plot(df)

    def test_sky_plot_data_summary(self, pulsar_df: pd.DataFrame) -> None:
        """Test sky plot data summary."""
        result = create_sky_plot(pulsar_df)

        assert "coordinate_system" in result.data_summary
        assert "n_pulsars" in result.data_summary
        assert "lon_range" in result.data_summary
        assert "lat_range" in result.data_summary


class TestComparisonPlot:
    """Tests for comparison plot creation."""

    def test_create_box_plot(self, pulsar_df: pd.DataFrame) -> None:
        """Test creating a box plot comparison."""
        result = create_comparison_plot(
            pulsar_df, parameter="P0", group_column="TYPE", plot_type="box"
        )

        assert result.figure is not None

    def test_create_violin_plot(self, pulsar_df: pd.DataFrame) -> None:
        """Test creating a violin plot comparison."""
        result = create_comparison_plot(
            pulsar_df, parameter="P0", group_column="TYPE", plot_type="violin"
        )

        assert result.figure is not None

    def test_comparison_missing_group(self, pulsar_df: pd.DataFrame) -> None:
        """Test comparison plot with missing group column."""
        with pytest.raises(ValueError, match="NONEXISTENT"):
            create_comparison_plot(
                pulsar_df, parameter="P0", group_column="NONEXISTENT"
            )

    def test_comparison_missing_parameter(self, pulsar_df: pd.DataFrame) -> None:
        """Test comparison plot with missing parameter."""
        with pytest.raises(ValueError, match="NONEXISTENT"):
            create_comparison_plot(
                pulsar_df, parameter="NONEXISTENT", group_column="TYPE"
            )

    def test_comparison_data_summary(self, pulsar_df: pd.DataFrame) -> None:
        """Test comparison plot data summary."""
        result = create_comparison_plot(
            pulsar_df, parameter="P0", group_column="TYPE"
        )

        assert "parameter" in result.data_summary
        assert "group_column" in result.data_summary
        assert "n_groups" in result.data_summary
        assert "groups" in result.data_summary
