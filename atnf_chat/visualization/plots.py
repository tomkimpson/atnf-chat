"""Visualization tools for pulsar data.

This module provides publication-quality visualizations including:
- P-Pdot diagram (Period vs Period Derivative)
- Histograms and distributions
- Scatter plots with correlations
- Sky distribution plots

All plots are generated using Plotly for interactivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class PlotResult:
    """Result from a plot generation function.

    Attributes:
        figure: Plotly figure object
        title: Plot title
        description: Description of what the plot shows
        data_summary: Summary of data used
    """

    figure: go.Figure
    title: str
    description: str
    data_summary: dict[str, Any]

    def to_html(self, include_plotlyjs: bool = True) -> str:
        """Convert figure to HTML string.

        Args:
            include_plotlyjs: Include Plotly.js library in HTML

        Returns:
            HTML string
        """
        return self.figure.to_html(
            include_plotlyjs="cdn" if include_plotlyjs else False,
            full_html=False,
        )

    def to_json(self) -> str:
        """Convert figure to JSON for frontend rendering."""
        return self.figure.to_json()


def create_pp_diagram(
    df: pd.DataFrame,
    highlight_groups: dict[str, str] | None = None,
    show_lines: bool = True,
    title: str = "Period - Period Derivative Diagram",
) -> PlotResult:
    """Create a P-Pdot diagram with classification regions.

    The P-Pdot diagram is the fundamental visualization for pulsar populations,
    showing period vs period derivative with lines of constant magnetic field
    and characteristic age.

    Args:
        df: DataFrame with P0 and P1 columns
        highlight_groups: Dict mapping group names to pandas query conditions
        show_lines: Whether to show constant B and age lines
        title: Plot title

    Returns:
        PlotResult with interactive Plotly figure

    Example:
        >>> result = create_pp_diagram(df, highlight_groups={
        ...     "MSPs": "P0 < 0.03",
        ...     "Magnetars": "BSURF > 1e14"
        ... })
        >>> result.figure.show()
    """
    if "P0" not in df.columns or "P1" not in df.columns:
        raise ValueError("DataFrame must contain P0 and P1 columns")

    fig = go.Figure()

    # Add constant characteristic age lines
    if show_lines:
        P_range = np.logspace(-3, 1.5, 100)

        # Lines of constant age
        age_lines = [
            (1e3, "1 kyr", "rgba(150, 150, 150, 0.5)"),
            (1e6, "1 Myr", "rgba(150, 150, 150, 0.5)"),
            (1e9, "1 Gyr", "rgba(150, 150, 150, 0.5)"),
        ]

        for age_yr, label, color in age_lines:
            # tau = P / (2 * Pdot) => Pdot = P / (2 * tau)
            age_seconds = age_yr * 365.25 * 24 * 3600
            Pdot = P_range / (2 * age_seconds)

            fig.add_trace(go.Scatter(
                x=P_range,
                y=Pdot,
                mode="lines",
                line=dict(color=color, dash="dash", width=1),
                name=f"τ = {label}",
                hoverinfo="name",
                showlegend=True,
            ))

        # Lines of constant magnetic field
        B_lines = [
            (1e8, "10⁸ G", "rgba(100, 149, 237, 0.4)"),
            (1e10, "10¹⁰ G", "rgba(100, 149, 237, 0.4)"),
            (1e12, "10¹² G", "rgba(100, 149, 237, 0.4)"),
            (1e14, "10¹⁴ G", "rgba(100, 149, 237, 0.4)"),
        ]

        for B_gauss, label, color in B_lines:
            # B = 3.2e19 * sqrt(P * Pdot) => Pdot = (B / 3.2e19)^2 / P
            Pdot = (B_gauss / 3.2e19) ** 2 / P_range

            fig.add_trace(go.Scatter(
                x=P_range,
                y=Pdot,
                mode="lines",
                line=dict(color=color, dash="dot", width=1),
                name=f"B = {label}",
                hoverinfo="name",
                showlegend=True,
            ))

        # Death line (approximate)
        death_Pdot = 1e-16 * (P_range / 1.0) ** 2
        fig.add_trace(go.Scatter(
            x=P_range,
            y=death_Pdot,
            mode="lines",
            line=dict(color="rgba(255, 0, 0, 0.3)", dash="dashdot", width=2),
            name="Death Line",
            hoverinfo="name",
            showlegend=True,
        ))

    # Plot pulsars
    if highlight_groups:
        for group_name, condition in highlight_groups.items():
            try:
                group_df = df.query(condition)
                _add_pulsar_scatter(fig, group_df, group_name)
            except Exception:
                # If query fails, skip this group
                continue
    else:
        _add_pulsar_scatter(fig, df, "Pulsars")

    # Layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Period (s)",
            type="log",
            range=[-3, 1.5],
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        yaxis=dict(
            title="Period Derivative",
            type="log",
            range=[-22, -9],
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
    )

    # Data summary
    valid_data = df[["P0", "P1"]].dropna()
    summary = {
        "total_pulsars": len(df),
        "plotted_pulsars": len(valid_data),
        "p0_range": [float(valid_data["P0"].min()), float(valid_data["P0"].max())],
        "p1_range": [float(valid_data["P1"].min()), float(valid_data["P1"].max())],
    }

    return PlotResult(
        figure=fig,
        title=title,
        description="Period-Period Derivative diagram showing pulsar populations "
                    "with lines of constant magnetic field and characteristic age.",
        data_summary=summary,
    )


def _add_pulsar_scatter(
    fig: go.Figure,
    df: pd.DataFrame,
    name: str,
) -> None:
    """Add pulsar scatter trace to figure."""
    valid = df[["P0", "P1"]].dropna()

    # Create hover text
    hover_text = []
    for idx, row in valid.iterrows():
        jname = df.loc[idx, "JNAME"] if "JNAME" in df.columns else f"PSR {idx}"
        text = f"<b>{jname}</b><br>P = {row['P0']:.4f} s<br>Ṗ = {row['P1']:.2e}"
        hover_text.append(text)

    fig.add_trace(go.Scatter(
        x=valid["P0"],
        y=valid["P1"],
        mode="markers",
        name=name,
        marker=dict(size=6, opacity=0.7),
        text=hover_text,
        hoverinfo="text",
    ))


def create_histogram(
    df: pd.DataFrame,
    parameter: str,
    nbins: int = 50,
    log_scale: bool = False,
    title: str | None = None,
) -> PlotResult:
    """Create a histogram for a pulsar parameter.

    Args:
        df: DataFrame with pulsar data
        parameter: Column name to plot
        nbins: Number of bins
        log_scale: Use logarithmic x-axis
        title: Plot title (auto-generated if None)

    Returns:
        PlotResult with histogram figure
    """
    if parameter not in df.columns:
        raise ValueError(f"Parameter {parameter} not found in DataFrame")

    data = df[parameter].dropna()

    if len(data) == 0:
        raise ValueError(f"No valid data for parameter {parameter}")

    if title is None:
        title = f"Distribution of {parameter}"

    fig = go.Figure()

    if log_scale and (data > 0).all():
        # Use log-spaced bins
        log_data = np.log10(data)
        bins = np.logspace(log_data.min(), log_data.max(), nbins + 1)
        fig.add_trace(go.Histogram(
            x=data,
            xbins=dict(start=bins[0], end=bins[-1], size=(bins[1] - bins[0])),
            name=parameter,
        ))
        fig.update_xaxes(type="log")
    else:
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=nbins,
            name=parameter,
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=parameter,
        yaxis_title="Count",
        template="plotly_white",
        bargap=0.05,
    )

    summary = {
        "parameter": parameter,
        "count": len(data),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
    }

    return PlotResult(
        figure=fig,
        title=title,
        description=f"Histogram of {parameter} distribution.",
        data_summary=summary,
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x_param: str,
    y_param: str,
    color_by: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
    title: str | None = None,
    show_regression: bool = False,
) -> PlotResult:
    """Create a scatter plot of two parameters.

    Args:
        df: DataFrame with pulsar data
        x_param: X-axis parameter
        y_param: Y-axis parameter
        color_by: Optional parameter for color coding
        log_x: Use log scale for x-axis
        log_y: Use log scale for y-axis
        title: Plot title
        show_regression: Show linear regression line

    Returns:
        PlotResult with scatter plot
    """
    required = [x_param, y_param]
    if color_by:
        required.append(color_by)

    for param in required:
        if param not in df.columns:
            raise ValueError(f"Parameter {param} not found in DataFrame")

    # Clean data
    plot_df = df[required].dropna()

    if len(plot_df) == 0:
        raise ValueError("No valid data for scatter plot")

    if title is None:
        title = f"{y_param} vs {x_param}"

    fig = go.Figure()

    # Create hover text
    hover_text = []
    for idx, row in plot_df.iterrows():
        jname = df.loc[idx, "JNAME"] if "JNAME" in df.columns else f"PSR {idx}"
        text = f"<b>{jname}</b><br>{x_param} = {row[x_param]:.4g}<br>{y_param} = {row[y_param]:.4g}"
        hover_text.append(text)

    if color_by:
        fig.add_trace(go.Scatter(
            x=plot_df[x_param],
            y=plot_df[y_param],
            mode="markers",
            marker=dict(
                size=6,
                color=plot_df[color_by],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_by),
            ),
            text=hover_text,
            hoverinfo="text",
            name="Pulsars",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df[x_param],
            y=plot_df[y_param],
            mode="markers",
            marker=dict(size=6, opacity=0.7),
            text=hover_text,
            hoverinfo="text",
            name="Pulsars",
        ))

    # Add regression line if requested
    if show_regression:
        x_data = plot_df[x_param].values
        y_data = plot_df[y_param].values

        if log_x:
            x_data = np.log10(x_data[x_data > 0])
            y_data = y_data[plot_df[x_param] > 0]
        if log_y:
            mask = y_data > 0
            x_data = x_data[mask]
            y_data = np.log10(y_data[mask])

        if len(x_data) > 2:
            coeffs = np.polyfit(x_data, y_data, 1)
            x_line = np.linspace(x_data.min(), x_data.max(), 100)
            y_line = np.polyval(coeffs, x_line)

            if log_x:
                x_line = 10 ** x_line
            if log_y:
                y_line = 10 ** y_line

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Fit (slope={coeffs[0]:.2f})",
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=x_param,
        yaxis_title=y_param,
        template="plotly_white",
        hovermode="closest",
    )

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    summary = {
        "x_param": x_param,
        "y_param": y_param,
        "n_points": len(plot_df),
    }

    return PlotResult(
        figure=fig,
        title=title,
        description=f"Scatter plot of {y_param} vs {x_param}.",
        data_summary=summary,
    )


def create_sky_plot(
    df: pd.DataFrame,
    color_by: str | None = None,
    projection: str = "mollweide",
    title: str = "Sky Distribution of Pulsars",
) -> PlotResult:
    """Create a sky distribution plot in Galactic coordinates.

    Args:
        df: DataFrame with GL (Galactic longitude) and GB (Galactic latitude)
        color_by: Optional parameter for color coding
        projection: Map projection ('mollweide' or 'equirectangular')
        title: Plot title

    Returns:
        PlotResult with sky map figure
    """
    # Check for required columns
    if "GL" not in df.columns or "GB" not in df.columns:
        # Try to use RAJD/DECJD if available
        if "RAJD" in df.columns and "DECJD" in df.columns:
            lon_col, lat_col = "RAJD", "DECJD"
            coord_system = "Equatorial (J2000)"
        else:
            raise ValueError("DataFrame must contain GL/GB or RAJD/DECJD columns")
    else:
        lon_col, lat_col = "GL", "GB"
        coord_system = "Galactic"

    required = [lon_col, lat_col]
    if color_by and color_by in df.columns:
        required.append(color_by)

    plot_df = df[required].dropna()

    if len(plot_df) == 0:
        raise ValueError("No valid coordinate data")

    fig = go.Figure()

    # Convert longitude to [-180, 180] for Galactic coordinates
    lon = plot_df[lon_col].values.copy()
    if lon_col == "GL":
        lon = np.where(lon > 180, lon - 360, lon)

    lat = plot_df[lat_col].values

    # Create hover text
    hover_text = []
    for idx, row in plot_df.iterrows():
        jname = df.loc[idx, "JNAME"] if "JNAME" in df.columns else f"PSR {idx}"
        text = f"<b>{jname}</b><br>{lon_col} = {row[lon_col]:.2f}°<br>{lat_col} = {row[lat_col]:.2f}°"
        hover_text.append(text)

    marker_kwargs = dict(size=4, opacity=0.7)

    if color_by and color_by in plot_df.columns:
        marker_kwargs["color"] = plot_df[color_by]
        marker_kwargs["colorscale"] = "Viridis"
        marker_kwargs["showscale"] = True
        marker_kwargs["colorbar"] = dict(title=color_by)

    fig.add_trace(go.Scattergeo(
        lon=lon,
        lat=lat,
        mode="markers",
        marker=marker_kwargs,
        text=hover_text,
        hoverinfo="text",
        name="Pulsars",
    ))

    # Configure projection
    if projection == "mollweide":
        geo_config = dict(
            projection_type="mollweide",
            showland=False,
            showocean=False,
            showlakes=False,
            showcountries=False,
            lonaxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
            lataxis=dict(showgrid=True, gridwidth=0.5, gridcolor="rgba(0,0,0,0.1)"),
            bgcolor="white",
        )
    else:
        geo_config = dict(
            projection_type="equirectangular",
            showland=False,
            bgcolor="white",
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        geo=geo_config,
        template="plotly_white",
    )

    summary = {
        "coordinate_system": coord_system,
        "n_pulsars": len(plot_df),
        "lon_range": [float(lon.min()), float(lon.max())],
        "lat_range": [float(lat.min()), float(lat.max())],
    }

    return PlotResult(
        figure=fig,
        title=title,
        description=f"Sky distribution of pulsars in {coord_system} coordinates.",
        data_summary=summary,
    )


def create_comparison_plot(
    df: pd.DataFrame,
    parameter: str,
    group_column: str,
    plot_type: str = "box",
    title: str | None = None,
) -> PlotResult:
    """Create a comparison plot between groups.

    Args:
        df: DataFrame with pulsar data
        parameter: Parameter to compare
        group_column: Column defining groups
        plot_type: 'box' or 'violin'
        title: Plot title

    Returns:
        PlotResult with comparison figure
    """
    if parameter not in df.columns:
        raise ValueError(f"Parameter {parameter} not found")
    if group_column not in df.columns:
        raise ValueError(f"Group column {group_column} not found")

    plot_df = df[[parameter, group_column]].dropna()

    if title is None:
        title = f"{parameter} by {group_column}"

    fig = go.Figure()

    groups = plot_df[group_column].unique()

    for group in groups:
        group_data = plot_df[plot_df[group_column] == group][parameter]

        if plot_type == "violin":
            fig.add_trace(go.Violin(
                y=group_data,
                name=str(group),
                box_visible=True,
                meanline_visible=True,
            ))
        else:
            fig.add_trace(go.Box(
                y=group_data,
                name=str(group),
                boxmean=True,
            ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title=parameter,
        xaxis_title=group_column,
        template="plotly_white",
    )

    summary = {
        "parameter": parameter,
        "group_column": group_column,
        "n_groups": len(groups),
        "groups": list(groups),
    }

    return PlotResult(
        figure=fig,
        title=title,
        description=f"Comparison of {parameter} across {group_column} groups.",
        data_summary=summary,
    )
