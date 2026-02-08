import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px
import folium
import warnings
import branca.colormap as cm
import plotly.graph_objects as go
from folium.plugins import Search
from scipy.stats import gaussian_kde

warnings.simplefilter(action='ignore', category=FutureWarning)


# ----------------------------
# Define Measure EDA Class
# ----------------------------
class SubdivisionMeasureEDA:
    def __init__(self, gdf, measure_col):
        self.gdf = gdf.copy()
        self.col = measure_col
        self.clean_gdf = self.gdf[self.gdf[self.col].notna()].copy()
        self.clean_gdf[f'{self.col}_log'] = np.log1p(self.clean_gdf[self.col])

    # ------------------------------
    # Summary statistics
    # ------------------------------
    def summary_stats(self):
        desc = self.gdf[self.col].describe()
        median_val = self.gdf[self.col].median()
        missing_count = self.gdf[self.col].isna().sum()
        missing_pct = missing_count / len(self.gdf) * 100
        print(f"Summary for '{self.col}':")
        print(desc)
        print(f"Median: {median_val:.4f}")
        print(f"Missing: {missing_count} ({missing_pct:.2f}%)")

    # ------------------------------
    # Missing value analysis
    # ------------------------------
    def missing_analysis(self):
        """
        Analyze missing values for the measure:
        - Counts and ratio per year (Plotly dual-axis chart)
        - Folium map of missing subdivisions
        - List of missing subdivision IDs and detailed table
        """

        # --- MISSING COUNT & RATIO ANALYSIS ---
        total_per_year = self.gdf.groupby('year').size().reset_index(name='total')

        missing_per_year = (
            self.gdf[self.gdf[self.col].isna()]
            .groupby('year')
            .size()
            .reset_index(name='missing_count')
        )

        ratio_df = pd.merge(total_per_year, missing_per_year, on='year', how='left')
        ratio_df['missing_count'] = ratio_df['missing_count'].fillna(0)
        ratio_df['missing_ratio'] = ratio_df['missing_count'] / ratio_df['total']

        # --- VISUALIZATION 1: Plotly Dual-Axis Chart ---
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ratio_df['year'],
            y=ratio_df['missing_count'],
            name='Missing Count',
            text=ratio_df['missing_count'].astype(int),
            textposition='outside',
            marker_color='blue'
        ))
        fig.add_trace(go.Scatter(
            x=ratio_df['year'],
            y=ratio_df['missing_count'],
            mode='lines+markers',
            name='Missing Count Trend',
            line=dict(color='blue', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=ratio_df['year'],
            y=ratio_df['missing_ratio'],
            mode='lines+markers+text',
            name='Missing Ratio',
            line=dict(color='red', width=2),
            yaxis='y2',
            text=(ratio_df['missing_ratio'] * 100).round(1).astype(str) + '%',
            textposition='top center'
        ))
        fig.update_layout(
            title=f'Missing {self.col} Counts and Ratio by Year',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Missing Count'),
            yaxis2=dict(
                title='Missing Ratio',
                overlaying='y',
                side='right',
                tickformat=".0%"
            ),
            legend=dict(x=0.01, y=0.99),
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )
        fig.show()

        # --- VISUALIZATION 2: Folium Map ---
        missing_gdf = self.gdf[self.gdf[self.col].isna()].copy()
        if missing_gdf.empty:
            print(f"✔ No missing {self.col} values found.")
            return
        missing_vis = missing_gdf.to_crs(epsg=4326)

        map_ = folium.Map(location=[35.2265, -80.8409], zoom_start=13, tiles="CartoDB positron")
        popup = folium.GeoJsonPopup(fields=["subd_id", "issue_date"], localize=True, labels=True,
                                    style="background-color: yellow;")

        folium.GeoJson(
            data=missing_vis[['subd_id', 'issue_date', 'geometry']].to_json(),
            name=f"Missing {self.col}",
            style_function=lambda feature: {"color": "gray", "weight": 1, "fillOpacity": 0.5},
            highlight_function=lambda feature: {"color": "black", "weight": 4, "fillOpacity": 0.8},
            popup=popup
        ).add_to(map_)

        search = Search(
            layer=list(map_._children.values())[-1],
            search_label="subd_id",
            placeholder="Search by Sub ID",
            collapsed=False
        )
        search.add_to(map_)

        display(map_)

    # ------------------------------
    # Yearly trends
    # ------------------------------
    def yearly_trends(self, year_col='year'):
        df = (
            self.clean_gdf
            .groupby(year_col)[self.col]
            .agg(['mean', 'median', 'std', 'min', 'max', 'count'])
            .reset_index()
        )
        return df

    # ------------------------------
    # Plot distribution
    # ------------------------------
    def plot_distribution(self, log_transform=False, bins=30):
        data = self.clean_gdf[self.col] if not log_transform else self.clean_gdf[f'{self.col}_log']
        plt.figure(figsize=(12, 5))
        sns.histplot(data, bins=bins, kde=True, color='skyblue')
        plt.title(f"{self.col}" + (" (Log-transformed)" if log_transform else ""))
        plt.show()

        plt.figure(figsize=(12, 3))
        sns.boxplot(x=data, color='lightgreen')
        plt.title(f"Boxplot of {self.col}" + (" (Log-transformed)" if log_transform else ""))
        plt.show()

    # ------------------------------
    # KDE plot
    # ------------------------------
    def plot_kde(self):
        x_range = np.linspace(self.clean_gdf[f'{self.col}_log'].min(),
                              self.clean_gdf[f'{self.col}_log'].max(), 500)
        kde = gaussian_kde(self.clean_gdf[f'{self.col}_log'])
        kde_values = kde(x_range)

        fig = px.histogram(self.clean_gdf, x=f'{self.col}_log', nbins=50, histnorm='probability density',
                           title=f"{self.col} (Log-transformed) with KDE")
        fig.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines', name='KDE',
                                 line=dict(color='orange', width=2)))
        fig.show()

    # ------------------------------
    # Outlier detection
    # ------------------------------
    def detect_outliers(self, method='iqr', factor=1.5):
        if method == 'iqr':
            Q1 = self.gdf[self.col].quantile(0.25)
            Q3 = self.gdf[self.col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            self.outliers = self.gdf[(self.gdf[self.col] < lower) | (self.gdf[self.col] > upper)]
        else:
            raise NotImplementedError(f"Method {method} not implemented")
        return self.outliers

    # ------------------------------
    # Outlier trend plot
    # ------------------------------
    def plot_outlier_trends(self, date_col='year'):
        if not hasattr(self, 'outliers'):
            raise ValueError("Outliers not detected yet. Run detect_outliers() first.")

        total_per_year = self.gdf.groupby(date_col).size().reset_index(name='total')
        outliers_per_year = self.outliers.groupby(date_col).size().reset_index(name='outlier_count')

        ratio_df = pd.merge(total_per_year, outliers_per_year, on=date_col, how='left').fillna(0)
        ratio_df['outlier_ratio'] = ratio_df['outlier_count'] / ratio_df['total']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ratio_df[date_col], y=ratio_df['outlier_count'],
            name='Outlier Count', text=ratio_df['outlier_count'], textposition='outside',
            marker_color='orange'
        ))
        fig.add_trace(go.Scatter(
            x=ratio_df[date_col], y=ratio_df['outlier_count'],
            mode='lines+markers', name='Outlier Count Trend',
            line=dict(color='orange', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=ratio_df[date_col], y=ratio_df['outlier_ratio'],
            mode='lines+markers+text', name='Outlier Ratio',
            line=dict(color='red', width=2), yaxis='y2',
            text=(ratio_df['outlier_ratio'] * 100).round(1).astype(str) + '%',
            textposition='top center'
        ))
        fig.update_layout(
            title=f'Outlier Counts and Ratio per {date_col} ({self.col})',
            xaxis=dict(title='Year'),
            yaxis=dict(title='Outlier Count'),
            yaxis2=dict(title='Outlier Ratio', overlaying='y', side='right', tickformat=".0%"),
            legend=dict(x=0.01, y=0.99, font=dict(size=10)),
            uniformtext_minsize=10, uniformtext_mode='hide'
        )
        fig.show()

    # ------------------------------
    # Folium outlier map
    # ------------------------------
    def folium_outlier_map(self, building_footprint_dataset=None, date_col='issue_date'):
        if not hasattr(self, 'outliers'):
            raise ValueError("Outliers not detected yet. Run detect_outliers() first.")

        outliers_vis = self.outliers.to_crs(epsg=4326)
        map_ = folium.Map(location=[35.2265, -80.8409], zoom_start=13, tiles="CartoDB positron")

        popup = folium.GeoJsonPopup(
            fields=["subd_id", self.col, date_col],
            localize=True,
            labels=True,
            style="background-color: yellow;"
        )

        subs = folium.GeoJson(
            outliers_vis[['subd_id', self.col, date_col, 'geometry']].to_json(),
            name="Subdivisions",
            style_function=lambda f: {"color": "green", "weight": 1, "fillOpacity": 0.3},
            highlight_function=lambda f: {"color": "black", "weight": 4, "fillOpacity": 0.6},
            popup=popup
        )
        subs.add_to(map_)

        if building_footprint_dataset is not None:
            buildings_in_outliers = gpd.sjoin(
                building_footprint_dataset, self.outliers, how="inner", predicate="within"
            )
            buildings_vis = buildings_in_outliers.to_crs(epsg=4326)
            folium.GeoJson(
                buildings_vis['geometry'].to_json(),
                name="Buildings",
                style_function=lambda f: {"color": "red", "weight": 1, "fillOpacity": 0.3},
                highlight_function=lambda f: {"color": "black", "weight": 4, "fillOpacity": 0.6}
            ).add_to(map_)

        search = Search(layer=subs, search_label="subd_id", placeholder="Search by Sub ID", collapsed=False)
        search.add_to(map_)

        return map_

    # ------------------------------
    # General folium map
    # ------------------------------
    def folium_map(self, highlight_outliers=True):
        gdf_vis = self.gdf.to_crs(epsg=4326)

        missing_vis = gdf_vis[gdf_vis[self.col].isna()]

        if highlight_outliers and hasattr(self, 'outliers'):
            outliers_vis = self.outliers.to_crs(epsg=4326)
            regular_vis = gdf_vis[
                ~gdf_vis.index.isin(outliers_vis.index) & gdf_vis[self.col].notna()
            ]
        else:
            outliers_vis = gpd.GeoDataFrame(columns=gdf_vis.columns, crs=gdf_vis.crs)
            regular_vis = gdf_vis[gdf_vis[self.col].notna()]

        if not regular_vis.empty:
            min_val, max_val = regular_vis[self.col].min(), regular_vis[self.col].max()
            colormap = cm.LinearColormap(['blue', 'green', 'yellow'], vmin=min_val, vmax=max_val)
        else:
            colormap = None

        map_ = folium.Map(location=[35.2265, -80.8409], zoom_start=13, tiles="CartoDB positron")

        # Regular values
        if not regular_vis.empty:
            folium.GeoJson(
                regular_vis[['subd_id', self.col, 'geometry']].to_json(),
                name='Regular',
                style_function=lambda f: {
                    'fillColor': colormap(f['properties'][self.col]) if colormap else 'blue',
                    'color': colormap(f['properties'][self.col]) if colormap else 'blue',
                    'weight': 0.5,
                    'fillOpacity': 0.6
                },
                highlight_function=lambda f: {'weight': 3, 'color': 'black', 'fillOpacity': 0.8},
                tooltip=folium.GeoJsonTooltip(fields=['subd_id', self.col], labels=True)
            ).add_to(map_)

            if colormap:
                colormap.caption = f'{self.col} (Regular values)'
                colormap.add_to(map_)

        # Outliers
        if highlight_outliers and not outliers_vis.empty:
            folium.GeoJson(
                outliers_vis[['subd_id', self.col, 'geometry']].to_json(),
                name='Outliers',
                style_function=lambda f: {'color': 'red', 'weight': 1, 'fillOpacity': 0.5},
                highlight_function=lambda f: {'color': 'black', 'weight': 4, 'fillOpacity': 0.8},
                tooltip=folium.GeoJsonTooltip(fields=['subd_id', self.col], labels=True)
            ).add_to(map_)

        # Missing
        if not missing_vis.empty:
            folium.GeoJson(
                missing_vis[['subd_id', 'geometry']].to_json(),
                name='Missing',
                style_function=lambda f: {'color': 'gray', 'weight': 1, 'fillOpacity': 0.5},
                highlight_function=lambda f: {'color': 'black', 'weight': 4, 'fillOpacity': 0.8},
                tooltip=folium.GeoJsonTooltip(fields=['subd_id'], labels=True)
            ).add_to(map_)

        folium.LayerControl(collapsed=False).add_to(map_)

        return map_


    # ------------------------------
    # Plot yearly index/measure trends (bar chart + optional line)
    # ------------------------------
    def plot_yearly_index(self, year_col='year', agg_func='mean', add_line=True):
        """
        Creates a bar chart (and optional line trend) of the measure by year.

        Parameters
        ----------
        year_col : str
            Column name representing the year.
        agg_func : str or function
            Aggregation method ('mean', 'median', 'sum', 'std', or custom function).
        add_line : bool
            If True, a line trend is added over the bar chart.

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        # ---- Check inputs ----
        if year_col not in self.gdf.columns:
            raise ValueError(f"Column '{year_col}' not found in GeoDataFrame.")

        if self.col not in self.gdf.columns:
            raise ValueError(f"Measure column '{self.col}' not found.")

        # ---- Aggregate by year ----
        df = (
            self.gdf
            .groupby(year_col)[self.col]
            .agg(agg_func)
            .reset_index()
            .sort_values(year_col)
        )

        df.rename(columns={self.col: "value"}, inplace=True)

        # ---- Create Figure ----
        fig = go.Figure()

        # Bar Chart
        fig.add_trace(go.Bar(
            x=df[year_col],
            y=df["value"],
            name=f"{self.col} ({agg_func})",
            text=df["value"].round(2),
            textposition='outside',
            marker_color='steelblue'
        ))

        # Optional Line Trend
        if add_line:
            fig.add_trace(go.Scatter(
                x=df[year_col],
                y=df["value"],
                mode='lines+markers',
                name="Trend",
                line=dict(color='darkblue', width=2)
            ))

        fig.update_layout(
            title=f"{self.col} — {agg_func.capitalize()} by Year",
            xaxis_title="Year",
            yaxis_title=f"{self.col} ({agg_func})",
            uniformtext_minsize=10,
            uniformtext_mode='hide'
        )

        fig.show()
        return fig


    def plot_yearly_index_with_band(self, year_col='year', agg_func='mean'):
        """
        Plots yearly aggregation (mean/median/sum/etc.) with a min–max shaded band.

        Parameters
        ----------
        year_col : str
            Column indicating the year.
        agg_func : str or function
            Aggregation applied to the measure (mean, median, etc.)

        Returns
        -------
        fig : plotly.graph_objects.Figure
        """

        if year_col not in self.gdf.columns:
            raise ValueError(f"Column '{year_col}' not found.")
        if self.col not in self.gdf.columns:
            raise ValueError(f"Measure column '{self.col}' not found.")

        # ---- Compute min, max, and aggregation ----
        df = (
            self.gdf
            .groupby(year_col)[self.col]
            .agg(['min', 'max', agg_func])
            .reset_index()
            .sort_values(year_col)
        )
        df.rename(columns={agg_func: 'value'}, inplace=True)

        # ---- Plot ----
        fig = go.Figure()

        # Min line (bottom of band)
        fig.add_trace(go.Scatter(
            x=df[year_col],
            y=df['min'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Max line (top of band) with fill
        fig.add_trace(go.Scatter(
            x=df[year_col],
            y=df['max'],
            mode='lines',
            fill='tonexty',            # fill the area between min and max
            name='Min–Max Range',
            line=dict(width=0),
            fillcolor='rgba(100, 149, 237, 0.25)'  # light steelblue band
        ))

        # Mean (or chosen aggregation)
        fig.add_trace(go.Scatter(
            x=df[year_col],
            y=df['value'],
            mode='lines+markers',
            name=f"{self.col} ({agg_func})",
            line=dict(color='steelblue', width=3)
        ))

        fig.update_layout(
            title=f"{self.col} — {agg_func.capitalize()} with Min–Max Band by Year",
            xaxis_title="Year",
            yaxis_title=f"{self.col}",
            hovermode="x unified"
        )

        fig.show()
        return fig

    # ----------------------------------------------
    # Full table of missing-value records
    # ----------------------------------------------
    def missing_table(self, sort_by=None, ascending=True, reset_index=True):
        """
        Returns a DataFrame of all rows where the measure is missing.
        Includes ALL attributes/columns.

        Parameters
        ----------
        sort_by : str or list, optional
            Column(s) to sort the missing table by.
        ascending : bool, optional
            Sorting order (default: True).
        reset_index : bool, optional
            If True, reset index for clean display.

        Returns
        -------
        DataFrame
            Table of missing-value subdivisions with full attributes.
        """

        missing_df = self.gdf[self.gdf[self.col].isna()].copy()

        if sort_by:
            missing_df = missing_df.sort_values(sort_by, ascending=ascending)

        if reset_index:
            missing_df = missing_df.reset_index(drop=True)

        print(f"\n📌 Missing entries for '{self.col}': {len(missing_df)} rows\n")
        display(missing_df)

        return missing_df


    def query_by_value_and_date(
            self,
            min_value: float | None = None,
            max_value: float | None = None,
            percentile: float | None = None,
            percentile_side: str = "top",  # NEW: "top" or "bottom"
            year: int | None = None,
            start_year: int | None = None,
            end_year: int | None = None,
            date_col: str = "year",
            include_geometry: bool = True,
            reset_index: bool = True
    ):
        """
        Query records by measure value range AND year/date range.

        Parameters
        ----------
        min_value : float, optional
            Minimum value threshold for the measure
        max_value : float, optional
            Maximum value threshold for the measure
        percentile : float, optional
            Percentile threshold (e.g., 95 = top 5%, 5 = bottom 5%).
            Overrides min_value / max_value if provided.
        percentile_side : str
            "top" or "bottom" — controls which tail of the distribution is returned.
        year : int, optional
            Single year to filter (e.g., 2003)
        start_year : int, optional
            Start year (inclusive)
        end_year : int, optional
            End year (inclusive)
        date_col : str
            Column representing year or date
        include_geometry : bool
            Whether to keep geometry column
        reset_index : bool
            Whether to reset index before returning

        Returns
        -------
        GeoDataFrame or DataFrame
            Filtered records
        """

        if date_col not in self.gdf.columns:
            raise ValueError(f"Date column '{date_col}' not found.")

        df = self.gdf.copy()

        # -------------------------
        # Drop missing values
        # -------------------------
        df = df[df[self.col].notna()]

        # -------------------------
        # Date filtering
        # -------------------------
        if year is not None:
            df = df[df[date_col] == year]
        else:
            if start_year is not None:
                df = df[df[date_col] >= start_year]
            if end_year is not None:
                df = df[df[date_col] <= end_year]

        # -------------------------
        # Value filtering
        # -------------------------
        if percentile is not None:
            if not (0 < percentile < 100):
                raise ValueError("percentile must be between 0 and 100")

            percentile_side = percentile_side.lower()
            if percentile_side not in {"top", "bottom"}:
                raise ValueError("percentile_side must be 'top' or 'bottom'")

            q = percentile / 100
            threshold = df[self.col].quantile(q)

            if percentile_side == "top":
                df = df[df[self.col] >= threshold]
            else:  # bottom
                df = df[df[self.col] <= threshold]

        else:
            if min_value is not None:
                df = df[df[self.col] >= min_value]
            if max_value is not None:
                df = df[df[self.col] <= max_value]

        # -------------------------
        # Clean output
        # -------------------------
        if not include_geometry and "geometry" in df.columns:
            df = df.drop(columns="geometry")

        if reset_index:
            df = df.reset_index(drop=True)

        print(
            f"📌 Query result: {len(df)} records "
            f"(Measure: '{self.col}', "
            f"Date filter: {year or f'{start_year}–{end_year}'}, "
            f"Percentile: {percentile_side if percentile else 'N/A'})"
        )

        display(df)

        return df

