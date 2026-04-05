"""
Metrics Visualization Module for Multi-Agent Simulation Analysis

This module provides a comprehensive set of publication-quality visualization functions
for analyzing simulation metrics across different conditions, episodes, and agents.

Key Features:
    - Centralized style configuration for consistent theming
    - Reusable functions for distribution analysis (density, boxplots, violin plots)
    - Evolution tracking across episodes with optional smoothing
    - Full customization of titles, labels, colors, and plot parameters
    - Publication-ready output for scientific posters and papers

Classes:
    StyleConfig: Centralized configuration for all visual parameters
    MetricsVisualizer: Main class providing all visualization methods

Example:
    >>> from metrics_visualization import StyleConfig, MetricsVisualizer
    >>> config = StyleConfig(
    ...     title_fontsize=14,
    ...     palette="Set2",
    ...     context="paper"
    ... )
    >>> viz = MetricsVisualizer(config=config)
    >>> fig = viz.density_plot(df, column='total_combined_reward_0', hue='emotion')
    >>> fig.savefig('density_plot.pdf', dpi=300, bbox_inches='tight')

Author: Generated for scientific publication
License: Open source
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union, Callable
from scipy.ndimage import uniform_filter1d
import warnings

warnings.filterwarnings('ignore')


class StyleConfig:
    """
    Centralized configuration for all visual parameters and styles.
    
    This class manages all aspects of plot styling including fonts, colors,
    sizes, and matplotlib parameters. It ensures consistency across all
    visualizations and allows easy customization for different output formats
    (posters, papers, presentations).
    
    Attributes:
        title_fontsize (int): Font size for plot titles
        label_fontsize (int): Font size for axis labels
        tick_fontsize (int): Font size for axis ticks
        legend_fontsize (int): Font size for legend text
        palette (str): Seaborn color palette name
        style (str): Seaborn plot style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        context (str): Seaborn context ('paper', 'notebook', 'talk', 'poster')
        figsize (Tuple[int, int]): Default figure size (width, height) in inches
        dpi (int): Resolution in dots per inch (use 300+ for publication)
        line_width (float): Line width for plots
        marker_size (float): Size of plot markers
        palette_list (List[str]): Custom color palette override
    
    Example:
        >>> config = StyleConfig(
        ...     title_fontsize=16,
        ...     palette="husl",
        ...     context="paper",
        ...     dpi=300,
        ...     figsize=(10, 6)
        ... )
        >>> viz = MetricsVisualizer(config=config)
    """
    
    def __init__(
        self,
        title_fontsize: int = 14,
        label_fontsize: int = 12,
        tick_fontsize: int = 10,
        legend_fontsize: int = 10,
        palette: str = "husl",
        style: str = "whitegrid",
        context: str = "paper",
        figsize: Tuple[int, int] = (12, 6),
        dpi: int = 100,
        line_width: float = 2.0,
        marker_size: float = 8.0,
        palette_list: Optional[List[str]] = None,
        rc_params: Optional[Dict] = None
    ):
        """
        Initialize style configuration.
        
        Args:
            title_fontsize: Font size for plot titles (default: 14)
            label_fontsize: Font size for axis labels (default: 12)
            tick_fontsize: Font size for axis tick labels (default: 10)
            legend_fontsize: Font size for legend text (default: 10)
            palette: Seaborn palette name. Options: 'husl', 'Set1', 'Set2', 'Set3',
                    'Paired', 'Dark2', 'muted', 'colorblind', 'pastel', etc.
                    See: https://seaborn.pydata.org/tutorial/color_palettes.html
            style: Plot background style. Options: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
            context: Seaborn context for scaling. Options: 'paper', 'notebook', 'talk', 'poster'
            figsize: Default figure size as (width, height) in inches
            dpi: Resolution in dots per inch. Use 300+ for publication quality
            line_width: Width of lines in plots (default: 2.0)
            marker_size: Size of markers in scatter plots (default: 8.0)
            palette_list: Optional custom list of colors to override palette
            rc_params: Additional matplotlib rcParams to customize
        """
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.palette = palette
        self.style = style
        self.context = context
        self.figsize = figsize
        self.dpi = dpi
        self.line_width = line_width
        self.marker_size = marker_size
        self.palette_list = palette_list
        self.rc_params = rc_params or {}
        
        # Apply styles
        self._apply_styles()
    
    def _apply_styles(self):
        """Apply all configured styles to matplotlib and seaborn."""
        # Set seaborn style and context
        sns.set_style(self.style)
        sns.set_context(self.context)
        
        # Set palette
        if self.palette_list:
            sns.set_palette(self.palette_list)
        else:
            sns.set_palette(self.palette)
        
        # Configure matplotlib parameters
        rc_dict = {
            'figure.dpi': self.dpi,
            'font.size': self.label_fontsize,
            'axes.labelsize': self.label_fontsize,
            'axes.titlesize': self.title_fontsize,
            'axes.linewidth': 1.2,
            'xtick.labelsize': self.tick_fontsize,
            'ytick.labelsize': self.tick_fontsize,
            'xtick.major.width': 1.0,
            'ytick.major.width': 1.0,
            'legend.fontsize': self.legend_fontsize,
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
        }
        rc_dict.update(self.rc_params)
        plt.rcParams.update(rc_dict)
    
    def to_dict(self) -> Dict:
        """
        Return configuration as dictionary.
        
        Returns:
            Dict containing all configuration parameters
        """
        return {
            'title_fontsize': self.title_fontsize,
            'label_fontsize': self.label_fontsize,
            'tick_fontsize': self.tick_fontsize,
            'legend_fontsize': self.legend_fontsize,
            'palette': self.palette,
            'style': self.style,
            'context': self.context,
            'figsize': self.figsize,
            'dpi': self.dpi,
            'line_width': self.line_width,
            'marker_size': self.marker_size,
        }
    
    def __repr__(self) -> str:
        return f"StyleConfig(palette={self.palette}, style={self.style}, context={self.context})"


class MetricsVisualizer:
    """
    Main visualization class for simulation metrics analysis.
    
    This class provides methods to create publication-quality visualizations
    for multi-agent simulation data. All plots can be extensively customized
    through the StyleConfig object and per-plot parameters.
    
    All plots are designed to be saved as high-resolution PNG or PDF files
    suitable for scientific posters and papers.
    
    Example:
        >>> config = StyleConfig(dpi=300, context="poster")
        >>> viz = MetricsVisualizer(config=config)
        >>> fig = viz.density_plot(
        ...     data=df,
        ...     column='total_combined_reward_0',
        ...     hue='emotion',
        ...     title='Reward Distribution by Emotional State'
        ... )
        >>> fig.savefig('reward_dist.pdf', dpi=300, bbox_inches='tight')
    """
    
    def __init__(self, config: Optional[StyleConfig] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: StyleConfig instance. If None, uses default configuration.
        """
        self.config = config or StyleConfig()
    
    # ========== DISTRIBUTION PLOTS ==========
    
    def density_plot(
        self,
        data: pd.DataFrame,
        column: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Density",
        hue_order: Optional[List] = None,
        figsize: Optional[Tuple[int, int]] = None,
        ax: Optional[plt.Axes] = None,
        fill: bool = True,
        alpha: float = 0.6,
        **kwargs
    ) -> plt.Figure:
        """
        Create a kernel density estimation (KDE) plot.
        
        Useful for comparing distributions across different conditions.
        Perfect for showing the shape and spread of metric distributions.
        
        Args:
            data: Input DataFrame
            column: Column name to plot
            hue: Optional column name for grouping by color
            title: Plot title. If None, auto-generated from column name
            xlabel: Label for x-axis. If None, uses column name
            ylabel: Label for y-axis (default: "Density")
            hue_order: Order of hue categories. If None, uses natural order
            figsize: Figure size (width, height). If None, uses config default
            ax: Matplotlib axes to plot on. If None, creates new figure
            fill: Whether to fill under the density curve (default: True)
            alpha: Transparency level 0-1 (default: 0.6)
            **kwargs: Additional arguments passed to sns.kdeplot()
        
        Returns:
            Figure object (can be saved with fig.savefig())
        
        Example:
            >>> fig = viz.density_plot(
            ...     data=df,
            ...     column='total_personal_reward_0',
            ...     hue='emotion',
            ...     title='Personal Reward Distribution by Emotion',
            ...     xlabel='Cumulative Personal Reward'
            ... )
            >>> fig.savefig('density.png', dpi=300, bbox_inches='tight')
        """
        figsize = figsize or self.config.figsize
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        # Set default title if not provided
        title = title or f"Distribution of {column}"
        xlabel = xlabel or column
        
        # Create KDE plot
        if hue:
            # Plot separate density for each hue category
            for hue_val in hue_order or sorted(data[hue].unique()):
                subset = data[data[hue] == hue_val][column].dropna()
                sns.kdeplot(
                    data=subset,
                    fill=fill,
                    alpha=alpha,
                    label=str(hue_val),
                    ax=ax,
                    **kwargs
                )
        else:
            sns.kdeplot(
                data=data[column].dropna(),
                fill=fill,
                alpha=alpha,
                ax=ax,
                **kwargs
            )
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        if hue and ax.get_legend() is None:
            ax.legend(title=hue, fontsize=self.config.legend_fontsize)
        elif hue:
            ax.get_legend().set_title(hue)
        
        plt.tight_layout()
        return fig
    
    def boxplot(
        self,
        data: pd.DataFrame,
        column: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        hue_order: Optional[List] = None,
        figsize: Optional[Tuple[int, int]] = None,
        ax: Optional[plt.Axes] = None,
        palette: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Create a boxplot to show distribution, quartiles, and outliers.
        
        Excellent for comparing distributions across multiple conditions
        and identifying statistical properties at a glance.
        
        Args:
            data: Input DataFrame
            column: Column name to plot (y-axis)
            hue: Optional column name for grouping (x-axis). If None, shows single box
            title: Plot title. If None, auto-generated
            xlabel: Label for x-axis. If None, uses hue name or 'Category'
            ylabel: Label for y-axis. If None, uses column name
            hue_order: Order of hue categories
            figsize: Figure size (width, height). If None, uses config default
            ax: Matplotlib axes to plot on. If None, creates new figure
            palette: Color palette override for this plot
            **kwargs: Additional arguments passed to sns.boxplot()
        
        Returns:
            Figure object
        
        Example:
            >>> fig = viz.boxplot(
            ...     data=df,
            ...     column='total_combined_reward_0',
            ...     hue='emotion',
            ...     title='Combined Reward by Emotional State',
            ...     ylabel='Cumulative Combined Reward'
            ... )
        """
        figsize = figsize or self.config.figsize
        palette = palette or self.config.palette
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        # Set default labels
        title = title or f"Distribution of {column}"
        ylabel = ylabel or column
        xlabel = xlabel or (hue if hue else "Category")
        
        # Create boxplot
        sns.boxplot(
            data=data,
            x=hue,
            y=column,
            hue=hue,
            order=hue_order,
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            **kwargs
        )
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        # Remove duplicate legend if hue is on x-axis
        if hue and ax.get_legend():
            ax.legend().remove()
        
        plt.tight_layout()
        return fig
    
    def violinplot(
        self,
        data: pd.DataFrame,
        column: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        hue_order: Optional[List] = None,
        figsize: Optional[Tuple[int, int]] = None,
        ax: Optional[plt.Axes] = None,
        palette: Optional[str] = None,
        split: bool = False,
        **kwargs
    ) -> plt.Figure:
        """
        Create a violin plot combining boxplot with KDE.
        
        Violin plots show the full distribution shape while preserving
        summary statistics. Ideal for publication-quality comparisons
        across multiple conditions.
        
        Args:
            data: Input DataFrame
            column: Column name to plot (y-axis)
            hue: Optional column name for grouping (x-axis)
            title: Plot title. If None, auto-generated
            xlabel: Label for x-axis
            ylabel: Label for y-axis. If None, uses column name
            hue_order: Order of hue categories
            figsize: Figure size (width, height)
            ax: Matplotlib axes. If None, creates new figure
            palette: Color palette override
            split: If True and hue has 2 categories, splits violins (default: False)
            **kwargs: Additional arguments passed to sns.violinplot()
        
        Returns:
            Figure object
        
        Example:
            >>> fig = viz.violinplot(
            ...     data=df,
            ...     column='gini_coef',
            ...     hue='see_emotions',
            ...     title='Gini Coefficient Distribution by Emotion Visibility',
            ...     ylabel='Gini Coefficient'
            ... )
        """
        figsize = figsize or self.config.figsize
        palette = palette or self.config.palette
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        title = title or f"Distribution of {column}"
        ylabel = ylabel or column
        xlabel = xlabel or (hue if hue else "Category")
        
        # Create violin plot
        sns.violinplot(
            data=data,
            x=hue,
            y=column,
            hue=hue,
            order=hue_order,
            hue_order=hue_order,
            palette=palette,
            split=split,
            ax=ax,
            **kwargs
        )
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        # Remove duplicate legend if hue is on x-axis
        if hue and ax.get_legend():
            ax.legend().remove()
        
        plt.tight_layout()
        return fig
    
    # ========== EVOLUTION PLOTS ==========
    
    def evolution_plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        hue_order: Optional[List] = None,
        figsize: Optional[Tuple[int, int]] = None,
        ax: Optional[plt.Axes] = None,
        palette: Optional[str] = None,
        smooth: bool = False,
        smoothing_window: int = 5,
        smooth_method: str = "uniform",
        confidence_interval: Optional[float] = None,
        markers: bool = True,
        **kwargs
    ) -> plt.Figure:
        """
        Create a line plot showing metric evolution across episodes/steps.
        
        Perfect for tracking performance improvements over time. Supports
        smoothing to reduce noise and optional confidence intervals.
        
        Args:
            data: Input DataFrame (should be sorted by x-axis column)
            x: Column name for x-axis (typically 'episode' or 'step')
            y: Column name for y-axis (metric to track)
            hue: Optional column for grouping lines by color
            title: Plot title. If None, auto-generated
            xlabel: Label for x-axis. If None, uses x column name
            ylabel: Label for y-axis. If None, uses y column name
            hue_order: Order of hue categories
            figsize: Figure size (width, height)
            ax: Matplotlib axes. If None, creates new figure
            palette: Color palette override
            smooth: Whether to apply smoothing (default: False)
            smoothing_window: Window size for smoothing (default: 5)
            smooth_method: Smoothing method - 'uniform' (moving average) or 'exponential'
            confidence_interval: Size of confidence interval (95 for 95% CI, None for none)
            markers: Whether to show markers on data points (default: True)
            **kwargs: Additional arguments passed to plt.plot()
        
        Returns:
            Figure object
        
        Example:
            >>> fig = viz.evolution_plot(
            ...     data=df_episodes,
            ...     x='episode',
            ...     y='efficiency_metric',
            ...     hue='see_emotions',
            ...     title='Learning Curve: Efficiency Over Episodes',
            ...     xlabel='Episode',
            ...     ylabel='Efficiency Metric',
            ...     smooth=True,
            ...     smoothing_window=10
            ... )
        """
        figsize = figsize or self.config.figsize
        palette = palette or self.config.palette
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        else:
            fig = ax.get_figure()
        
        title = title or f"{y} Evolution Over {x}"
        xlabel = xlabel or x
        ylabel = ylabel or y
        
        # Get color palette
        colors = sns.color_palette(palette)
        
        # Plot lines for each group
        if hue:
            hue_values = hue_order or sorted(data[hue].unique())
            for idx, hue_val in enumerate(hue_values):
                subset = data[data[hue] == hue_val].sort_values(x)
                
                y_vals = subset[y].values
                x_vals = subset[x].values
                
                # Apply smoothing if requested
                if smooth:
                    y_vals = self._smooth_data(
                        y_vals,
                        window=smoothing_window,
                        method=smooth_method
                    )
                
                # Plot line
                color = colors[idx % len(colors)]
                marker = 'o' if markers else 'None'
                ax.plot(
                    x_vals,
                    y_vals,
                    label=str(hue_val),
                    color=color,
                    marker=marker,
                    linewidth=self.config.line_width,
                    markersize=self.config.marker_size if markers else 0,
                    **kwargs
                )
                
                # Add confidence interval if requested
                if confidence_interval and len(subset) > 1:
                    ci = confidence_interval / 100
                    std_err = self._calculate_rolling_std(subset[y].values, smoothing_window)
                    ax.fill_between(
                        x_vals,
                        y_vals - 1.96 * std_err,
                        y_vals + 1.96 * std_err,
                        alpha=0.2,
                        color=color
                    )
            
            ax.legend(title=hue, fontsize=self.config.legend_fontsize)
        else:
            data_sorted = data.sort_values(x)
            y_vals = data_sorted[y].values
            x_vals = data_sorted[x].values
            
            if smooth:
                y_vals = self._smooth_data(
                    y_vals,
                    window=smoothing_window,
                    method=smooth_method
                )
            
            marker = 'o' if markers else 'None'
            ax.plot(
                x_vals,
                y_vals,
                color=colors[0],
                marker=marker,
                linewidth=self.config.line_width,
                markersize=self.config.marker_size if markers else 0,
                **kwargs
            )
            
            if confidence_interval:
                std_err = self._calculate_rolling_std(y_vals, smoothing_window)
                ax.fill_between(
                    x_vals,
                    y_vals - 1.96 * std_err,
                    y_vals + 1.96 * std_err,
                    alpha=0.2,
                    color=colors[0]
                )
        
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def evolution_faceted(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        facet_by: str,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        facet_order: Optional[List] = None,
        hue_order: Optional[List] = None,
        figsize: Optional[Tuple[int, int]] = None,
        smooth: bool = False,
        smoothing_window: int = 5,
        smooth_method: str = "uniform",
        **kwargs
    ) -> plt.Figure:
        """
        Create faceted line plots showing evolution across conditions.
        
        Useful for comparing multiple conditions side-by-side, where each
        condition gets its own subplot showing the evolution over episodes.
        
        Args:
            data: Input DataFrame
            x: Column name for x-axis (typically 'episode')
            y: Column name for y-axis (metric to track)
            facet_by: Column name to create separate subplots for each value
            hue: Optional column for coloring lines within each facet
            title: Overall title for the figure
            xlabel: Label for x-axes
            ylabel: Label for y-axes
            facet_order: Order for facet subplots
            hue_order: Order for hue categories
            figsize: Figure size (width, height)
            smooth: Whether to apply smoothing
            smoothing_window: Window size for smoothing
            smooth_method: 'uniform' or 'exponential'
            **kwargs: Additional arguments for plotting
        
        Returns:
            Figure object with subplots
        
        Example:
            >>> fig = viz.evolution_faceted(
            ...     data=df_episodes,
            ...     x='episode',
            ...     y='efficiency_metric',
            ...     facet_by='see_emotions',
            ...     hue='alpha',
            ...     title='Learning Curves by Emotion Visibility',
            ...     smooth=True,
            ...     smoothing_window=10
            ... )
        """
        figsize = figsize or (15, 6)
        
        facet_values = facet_order or sorted(data[facet_by].unique())
        n_facets = len(facet_values)
        
        fig, axes = plt.subplots(
            1,
            n_facets,
            figsize=figsize,
            dpi=self.config.dpi,
            sharey=True
        )
        
        if n_facets == 1:
            axes = [axes]
        
        title = title or f"{y} Evolution by {facet_by}"
        xlabel = xlabel or x
        ylabel = ylabel or y
        
        colors = sns.color_palette(self.config.palette)
        
        for idx, facet_val in enumerate(facet_values):
            ax = axes[idx]
            subset = data[data[facet_by] == facet_val]
            
            if hue:
                hue_values = hue_order or sorted(subset[hue].unique())
                for h_idx, hue_val in enumerate(hue_values):
                    h_subset = subset[subset[hue] == hue_val].sort_values(x)
                    
                    y_vals = h_subset[y].values
                    x_vals = h_subset[x].values
                    
                    if smooth:
                        y_vals = self._smooth_data(y_vals, smoothing_window, smooth_method)
                    
                    color = colors[h_idx % len(colors)]
                    ax.plot(
                        x_vals,
                        y_vals,
                        label=str(hue_val),
                        color=color,
                        linewidth=self.config.line_width,
                        marker='o',
                        markersize=self.config.marker_size,
                        **kwargs
                    )
                
                if idx == 0:
                    ax.legend(title=hue, fontsize=self.config.legend_fontsize - 2)
            else:
                subset_sorted = subset.sort_values(x)
                y_vals = subset_sorted[y].values
                x_vals = subset_sorted[x].values
                
                if smooth:
                    y_vals = self._smooth_data(y_vals, smoothing_window, smooth_method)
                
                ax.plot(
                    x_vals,
                    y_vals,
                    color=colors[0],
                    linewidth=self.config.line_width,
                    marker='o',
                    markersize=self.config.marker_size,
                    **kwargs
                )
            
            ax.set_title(f"{facet_by}: {facet_val}", fontsize=self.config.label_fontsize)
            ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize - 1)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        fig.suptitle(title, fontsize=self.config.title_fontsize, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig
    
    # ========== UTILITY FUNCTIONS ==========
    
    @staticmethod
    def _smooth_data(
        data: np.ndarray,
        window: int = 5,
        method: str = "uniform"
    ) -> np.ndarray:
        """
        Smooth data using moving average or exponential smoothing.
        
        Args:
            data: Input data array
            window: Window size for smoothing
            method: 'uniform' for moving average, 'exponential' for exponential smoothing
        
        Returns:
            Smoothed data array
        """
        if len(data) < window:
            return data
        
        if method == "uniform":
            return uniform_filter1d(data, size=window, mode='nearest')
        elif method == "exponential":
            alpha = 2 / (window + 1)
            smoothed = np.zeros_like(data, dtype=float)
            smoothed[0] = data[0]
            for i in range(1, len(data)):
                smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
            return smoothed
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    @staticmethod
    def _calculate_rolling_std(data: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Calculate rolling standard error for confidence intervals.
        
        Args:
            data: Input data array
            window: Window size
        
        Returns:
            Rolling standard error array
        """
        std_err = np.zeros_like(data, dtype=float)
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            std_err[i] = np.std(data[start:end]) / np.sqrt(end - start)
        return std_err
    
    def save_figure(
        self,
        fig: plt.Figure,
        filepath: str,
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
        **kwargs
    ) -> None:
        """
        Save figure with publication-quality settings.
        
        Args:
            fig: Matplotlib figure object
            filepath: Output file path (supports .png, .pdf, .eps, etc.)
            dpi: Resolution (default: uses config.dpi)
            bbox_inches: Bounding box setting (default: 'tight')
            **kwargs: Additional arguments for savefig()
        
        Example:
            >>> fig = viz.density_plot(df, 'total_combined_reward_0')
            >>> viz.save_figure(fig, 'output.pdf', dpi=300)
        """
        dpi = dpi or self.config.dpi
        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **kwargs
        )
        print(f"Figure saved to: {filepath}")


# ========== CONVENIENCE FUNCTIONS ==========

def create_default_config(context: str = "paper") -> StyleConfig:
    """
    Create a default StyleConfig optimized for publications.
    
    Args:
        context: Output context - 'paper', 'poster', 'presentation', 'notebook'
    
    Returns:
        Configured StyleConfig instance
    """
    configs = {
        "paper": StyleConfig(
            title_fontsize=14,
            label_fontsize=11,
            tick_fontsize=10,
            legend_fontsize=10,
            context="paper",
            dpi=300,
            figsize=(10, 6)
        ),
        "poster": StyleConfig(
            title_fontsize=18,
            label_fontsize=14,
            tick_fontsize=12,
            legend_fontsize=12,
            context="poster",
            dpi=300,
            figsize=(14, 8)
        ),
        "presentation": StyleConfig(
            title_fontsize=16,
            label_fontsize=13,
            tick_fontsize=11,
            legend_fontsize=11,
            context="talk",
            dpi=150,
            figsize=(12, 7)
        ),
        "notebook": StyleConfig(
            title_fontsize=14,
            label_fontsize=12,
            tick_fontsize=10,
            legend_fontsize=10,
            context="notebook",
            dpi=100,
            figsize=(12, 6)
        ),
    }
    return configs.get(context, configs["paper"])


if __name__ == "__main__":
    print("Metrics Visualization Module")
    print("=" * 50)
    print("Usage example:")
    print()
    print("from metrics_visualization import StyleConfig, MetricsVisualizer")
    print()
    print("config = StyleConfig(dpi=300, context='poster')")
    print("viz = MetricsVisualizer(config=config)")
    print()
    print("fig = viz.density_plot(")
    print("    data=df,")
    print("    column='total_combined_reward_0',")
    print("    hue='emotion'")
    print(")")
    print("fig.savefig('output.pdf', dpi=300, bbox_inches='tight')")