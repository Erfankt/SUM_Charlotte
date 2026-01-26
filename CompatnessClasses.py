#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import mapping
from shapely.ops import unary_union, transform
from scipy.ndimage import convolve, label
from pyproj import Transformer


"""
Compute BOTH sets of compactness metrics for each subdivision polygon:

--------------------------------------------------------------------------------
(A) SHAPE COMPACTNESS (Subdivision Boundary-Level Metrics)
--------------------------------------------------------------------------------
These metrics evaluate how efficient, regular, or compact the *overall*
subdivision boundary is. They help quantify whether the subdivision shape is
simple and compact (e.g., circular or rectangular) or irregular and sprawling
with excessive edge length relative to its area.

1. PARA (Perimeter–Area Ratio)
   - Basic compactness measure: larger values indicate more perimeter relative
     to area, meaning the subdivision boundary is more irregular.

2. CIRCLE (Circularity Ratio)
   - Compares the subdivision to a perfect circle of the same perimeter.
     Values closer to 1 indicate highly compact, circle-like shapes.

3. FRAC_SUB (Fractal Dimension of Subdivision Boundary)
   - Captures the complexity of the boundary across multiple scales. Higher
     values indicate increasingly convoluted or jagged boundaries.

4. SHAPE_SUB (FRAGSTATS Shape Index, Subdivision Level)
   - Standard FRAGSTATS shape metric expressing how much a polygon deviates
     from a compact shape. Index increases as the subdivision becomes less
     compact and more elongated or irregular.

5. CORE_AREA
   - Computes the interior area remaining after buffering the boundary inward
     (default = 10 ft). Lower core area values indicate thin or highly
     indented shapes with limited interior “bulk.”

--------------------------------------------------------------------------------
(B) CONTENT COMPACTNESS (Building Footprints Inside Subdivision)
--------------------------------------------------------------------------------
These metrics evaluate how buildings are arranged *within* the subdivision—
whether they are clustered, dispersed, cohesive, or fragmented. Together,
they describe the internal spatial structure of development.

6. AI (Aggregation Index)
   - Indicates the degree to which buildings are aggregated into clusters rather
     than isolated. Higher values indicate stronger clustering.

7. CLUMPY (Clumpiness Index)
   - Measures the tendency of buildings to form contiguous clumps. High values
     reflect compact groups of buildings rather than scattered development.

8. COHESION (Cohesion Index)
   - Evaluates how physically connected or contiguous the built pattern is.
     Higher cohesion indicates that buildings form a more unified, less
     fragmented arrangement.

9. ENN_MN (Mean Euclidean Nearest-Neighbor Distance)
   - Average straight-line distance between each building patch and its nearest
     neighboring patch, computed in raster space using FRAGSTATS-style
     edge-to-edge distances between patch cell centers.

10. PROX (Proximity Index)
    - Quantifies both distance and area of neighboring buildings within a
      specified search radius. Higher values indicate denser and more proximate
      clusters of buildings.

11. ED (Edge Density)
    - Total length of building edges per unit area (m/ha). Implemented as
      FRAGSTATS C5 class-level Edge Density for the building class.

12. LSI (Landscape Shape Index)
    - FRAGSTATS metric describing overall shape complexity of the building
      pattern. Higher values reflect more fragmented or irregular building
      distributions.

13. SHAPE_MN (Mean Building Shape Index)
    - Average shape complexity across all buildings. Higher values represent
      more irregular individual building footprints.

14. FRAC_MN (Mean Building Fractal Dimension)
    - Average fractal dimension of building footprints. Higher values indicate
      more complex or indented building shapes.

--------------------------------------------------------------------------------
EXCLUDED FROM THIS MODULE
--------------------------------------------------------------------------------
- FAR and BAD (computed in separate workflow components)
"""


# ================================================================
# COMPACTNESS ENGINE (Subdivision + Buildings)
# ===========================================resolution=====================
class CompactnessEngine:
    """
    Compute shape compactness (subdivision-level)
    and content compactness (buildings-as-patches).

    Assumes all geometries are in a US-feet CRS (e.g., EPSG:2264).
    """

    def __init__(self, buildings_gdf: gpd.GeoDataFrame):
        # Ensure coordinate system exists (must be projected)
        if buildings_gdf.crs is None:
            raise ValueError("Buildings GeoDataFrame must have a CRS.")

        self.buildings = buildings_gdf
        self.sindex = buildings_gdf.sindex   # speed up spatial queries
        self.buildings_union = self.buildings.unary_union

        # For some metrics (fractal dimension), we need meters
        self.to_meters = Transformer.from_crs(
            buildings_gdf.crs,  # US-feet CRS
            "EPSG:32119", # StatePlane NC meters
            always_xy=True
        )

    def _rasterize_buildings(self, geoms, subdivision, resolution):
        minx, miny, maxx, maxy = subdivision.bounds
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

        transform_aff = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
        shapes = ((mapping(g), 1) for g in geoms if not g.is_empty)

        return features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform_aff,
            fill=0,
            dtype=np.uint8
        )

    # ================================================================
    # (A) SHAPE COMPACTNESS -- subdivision geometry only
    # ================================================================
    @staticmethod
    def perimeter_area_ratio(subd):
        # Higher PARA → more elongated or irregular
        return 0.0 if subd.area <= 0 else float(subd.length / subd.area)

    @staticmethod
    def circularity_ratio(subd):
        # 1.0 = perfect circle (max compactness)
        A, P = subd.area, subd.length
        return 0.0 if A <= 0 or P <= 0 else float((4 * np.pi * A) / (P ** 2))

    def fractal_dimension_subd(self, subd):
        """
        FRAGSTATS fractal dimension for subdivision (FD).
        Uses perimeter–area scaling in meters for numerical stability.
        """

        # --- 1. Reproject subdivision to meters
        subd_m = transform(self.to_meters.transform, subd)
        subd_m = transform(self.to_meters.transform, subd)

        A = subd_m.area
        P = subd_m.length

        # --- 2. Prevent invalid values
        if A <= 1e-6 or P <= 1e-6:
            return 0.0

        # --- 3. FRAGSTATS formula
        FD = 2 * np.log(max(0.25 * P, 1e-6)) / np.log(max(A, 1e-6))

        # --- 4. Clamp to realistic range
        if FD < 0 or FD > 3:
            return np.nan

        return float(FD)

    @staticmethod
    def shape_index_subd(subd):
        # FRAGSTATS analog: compares shape to a perfect square
        A = subd.area
        return 0.0 if A <= 0 else max(1, float(0.25 * subd.length / np.sqrt(A)))

    @staticmethod
    def core_area(subd, buffer_dist=10):
        """
        Remove edges to get "core" (thick interior).
        buffer_dist is in feet (environment is US feet).
        """
        core = subd.buffer(-buffer_dist)
        return 0.0 if core.is_empty else float(core.area)

    # ================================================================
    # (B) CONTENT COMPACTNESS -- treat buildings as patches
    # ================================================================
    def _clip_buildings(self, subdivision, min_area_ft2=200):
        """
        Return building pieces inside the subdivision,
        excluding tiny clipped fragments (< min_area_ft2).
        """

        # Spatial index prefilter
        idx = list(self.sindex.intersection(subdivision.bounds))
        subset = self.buildings.iloc[idx].copy()

        # True geometric intersection
        subset = subset[subset.intersects(subdivision)]
        subset["geometry"] = subset.geometry.intersection(subdivision)

        # Drop empty geometries
        subset = subset[~subset.geometry.is_empty]

        # Explode multipart geometries (important after clipping)
        subset = subset.explode(index_parts=False)

        # Remove tiny sliver fragments
        subset = subset[subset.geometry.area >= min_area_ft2]

        return subset

    # ------------------------ SHAPE METRICS -------------------------
    @staticmethod
    def building_shape_index(geom):
        return 0.0 if geom.area == 0 else float(0.25 * geom.length / np.sqrt(geom.area))

    @staticmethod
    def _safe_log(x):
        return np.log(max(x, 1e-6))

    def building_fractal_dimension(self, geom):
        """
        Compute FRAGSTATS fractal dimension of a building polygon.
        Geometry is reprojected to meters to avoid log(A < 1 ft²) errors.
        """

        # --- Convert to meters ---
        geom_m = transform(self.to_meters.transform, geom)

        A = geom_m.area
        P = geom_m.length

        # --- Guard against tiny or degenerate polygons ---
        if A <= 1e-6 or P <= 1e-6:
            return np.nan

        def safelog(x):
            return np.log(max(x, 1e-6))

        # --- FRAGSTATS fractal dimension formula ---
        FD = 2 * safelog(0.25 * P) / safelog(A)

        # --- Clamp to realistic values (1.0–2.0 for buildings) ---
        if FD < 0 or FD > 3:
            return np.nan

        return float(FD)

    # ----------------- NEAREST-NEIGHBOR DISTANCES -------------------
    @staticmethod
    def mean_nearest_neighbor(geoms):
        """
        Centroid-based nearest-neighbor distances (not used for ENN_MN anymore).
        Kept as a helper in case you need vector NN elsewhere.
        """
        if len(geoms) < 2:
            return 0.0

        centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms])
        dmat = np.sqrt(((centroids[:, None] - centroids[None, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dmat, np.inf)

        return float(dmat.min(axis=1).mean())

    def enn_mn_raster(self, raster, subdivision, resolution=10):
        """
        TRUE FRAGSTATS-STYLE ENN_MN (patch-based, edge-to-edge, raster mode).

        Steps:
        1. Label building patches in the binary raster (4-neighbor connectivity).
        2. For each patch, compute the minimum Euclidean distance (in CRS units)
           between any cell center in that patch and any cell center in every
           other patch (edge-to-edge in raster space).
        3. ENN_MN is the mean of these nearest neighbor distances across patches.
        """

        # Binary mask of building cells
        mask = (raster == 1)
        if mask.sum() <= 1:
            return 0.0

        # 4-neighbor connectivity (rook adjacency)
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

        labeled, num_patches = label(mask, structure=structure)

        # If fewer than 2 patches
        if num_patches < 2:
            return 1.0

        # Map each patch ID to its list of cell-center coordinates
        patch_points = {pid: [] for pid in range(1, num_patches + 1)}

        rows, cols = np.where(labeled > 0)
        minx, miny, maxx, maxy = subdivision.bounds

        for r, c in zip(rows, cols):
            pid = labeled[r, c]
            # cell center X/Y in feet
            x = minx + (c + 0.5) * resolution
            y = maxy - (r + 0.5) * resolution
            patch_points[pid].append((x, y))

        # Convert lists to numpy arrays
        for pid in patch_points:
            patch_points[pid] = np.array(patch_points[pid])

        enn_values = []

        # For each focal patch, compute nearest neighbor distance
        for focal_pid, focal_xy in patch_points.items():

            if focal_xy.shape[0] == 0:
                continue

            min_dist = np.inf

            for other_pid, other_xy in patch_points.items():
                if other_pid == focal_pid or other_xy.shape[0] == 0:
                    continue

                diff = focal_xy[:, None, :] - other_xy[None, :, :]
                dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

                candidate = dist_matrix.min()
                if candidate < min_dist:
                    min_dist = candidate

            if np.isfinite(min_dist):
                h_i = (1.0 + min_dist) / 2.0
                enn_values.append(h_i)

        if len(enn_values) == 0:
            return 0.0

        # Mean ENN across all patches (still in feet)
        return float(np.mean(enn_values))

    # ------------------------ PROXIMITY INDEX -----------------------
    def proximity_index(self, geoms, subdivision,
                        resolution=10, search_radius=100, epsilon=1.0):
        """
        Proximity Index (PROX) following FRAGSTATS formulation,
        with explicit inclusion of the focal building using an
        arbitrarily small self-distance (epsilon = 1 ft).

        PROX_i = (a_i / 1) + sum_j (a_j / h_ij^2)
        Subdivision PROX = mean(PROX_i)
        """

        # 1. Rasterize buildings
        raster = self._rasterize_buildings(geoms, subdivision, resolution)

        # FRAGSTATS 4-connectivity
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])

        labeled, num_patches = label(raster == 1, structure=structure)

        if num_patches == 0:
            return 0.0

        # 2. Extract raster cell centers for each patch
        patch_cells = {i: [] for i in range(1, num_patches + 1)}

        rows, cols = np.where(labeled > 0)
        minx, miny, maxx, maxy = subdivision.bounds

        for r, c in zip(rows, cols):
            pid = labeled[r, c]
            x = minx + (c + 0.5) * resolution
            y = maxy - (r + 0.5) * resolution
            patch_cells[pid].append((x, y))

        # Patch areas (ft²)
        patch_areas = {
            pid: len(cells) * (resolution ** 2)
            for pid, cells in patch_cells.items()
        }

        # 3. Compute PROX for each focal building
        prox_values = []

        for focal_pid, focal_cells in patch_cells.items():
            if not focal_cells:
                continue

            focal_xy = np.array(focal_cells)

            # ---- SELF TERM: a_i / 1 ----
            prox_i = patch_areas[focal_pid] / (epsilon ** 2)

            # ---- NEIGHBOR TERMS ----
            for other_pid, other_cells in patch_cells.items():
                if other_pid == focal_pid or not other_cells:
                    continue

                other_xy = np.array(other_cells)

                dist_matrix = np.sqrt(
                    ((focal_xy[:, None] - other_xy[None, :]) ** 2).sum(axis=2)
                )

                h_edge = dist_matrix.min()

                if h_edge <= search_radius:
                    prox_i += patch_areas[other_pid] / (h_edge ** 2)

            prox_values.append(prox_i)

        return float(np.mean(prox_values)) if prox_values else 0.0

    # ----------------------- EDGE-BASED METRICS ----------------------
    def edge_density_raster(self, raster, subdivision, resolution_feet):
        """
        TRUE FRAGSTATS CLASS-LEVEL EDGE DENSITY (C5) for the building class.

        Environment CRS units: US feet.
        ED is reported in meters per hectare (m/ha), using:

            ED = (e_i / A) * 10,000

        where:
            e_i = total building edge length (meters) along the boundary
                  between building (1) and background (0) cells (4-neighbor),
            A   = subdivision area (m²).
        """

        # --- 1. Convert raster resolution from feet → meters ---
        res_m = resolution_feet * 0.3048

        # --- 2. Convert subdivision area from ft² → m² ---
        area_ft2 = subdivision.area
        area_m2 = area_ft2 * (0.3048 ** 2)

        if area_m2 <= 0:
            return 0.0

        rows, cols = raster.shape
        edge_length_m = 0.0

        # --- 3. Count edges of building cells against background (4-neighbor) ---
        for r in range(rows):
            for c in range(cols):
                if raster[r, c] == 1:

                    # up
                    if r == 0 or raster[r - 1, c] == 0:
                        edge_length_m += res_m
                    # down
                    if r == rows - 1 or raster[r + 1, c] == 0:
                        edge_length_m += res_m
                    # left
                    if c == 0 or raster[r, c - 1] == 0:
                        edge_length_m += res_m
                    # right
                    if c == cols - 1 or raster[r, c + 1] == 0:
                        edge_length_m += res_m

        # --- 4. FRAGSTATS Edge Density formula (m/ha) ---
        ED = (edge_length_m / area_m2) * 10000.0
        return float(ED)

    # ------------------ AGGREGATION / CLUMPY / COHESION --------------
    @staticmethod
    def aggregation_index(raster):
        """
        True FRAGSTATS Aggregation Index (AI) for a binary raster of buildings (1) and background (0).
        Returns a proportion in [0,1].
        """

        binrast = (raster == 1).astype(int)

        N = binrast.sum()
        if N <= 1:
            return 0.0

        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        adj = convolve(binrast, kernel, mode="constant", cval=0)
        gii = (adj * binrast).sum() / 2

        n = int(np.floor(np.sqrt(N)))
        m = N - n * n

        if m == 0:
            gii_max = 2 * n * (n - 1)
        elif m <= n:
            gii_max = 2 * n * (n - 1) + (2 * m - 1)
        elif m > n:
            gii_max = 2 * n * (n - 1) + (2 * m - 2)

        if gii_max <= 0:
            return 0.0

        return float(gii / gii_max)

    @staticmethod
    def clumpiness(raster):
        """
        True FRAGSTATS CLUMPY for a binary class (buildings=1, background=0).
        Returns float in [-1, 1]
        """

        raster = raster.astype(int)
        p_i = raster.mean()
        if p_i == 0:
            return 0.0

        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        adj = convolve(raster, kernel, mode='constant', cval=0)
        g_ii = (adj[raster == 1]).sum()

        max_adj = 4 * (raster == 1).sum()
        g_i = p_i * max_adj

        numerator = g_ii - g_i
        denominator = max_adj - g_i

        if denominator == 0:
            return 0.0

        clumpy = numerator / denominator
        return float(max(-1, min(1, clumpy)))


    # ================================================================
    # MAIN PIPELINE — compute all metrics for one subdivision
    # ================================================================
    def compute_for_subdivision(self, subdivision, resolution=10):

        # ------------------- SHAPE METRICS ---------------------------
        shape_metrics = {
            "PARA":      self.perimeter_area_ratio(subdivision),
            "CIRCLE":    self.circularity_ratio(subdivision),
            "FRAC_SUB":  self.fractal_dimension_subd(subdivision),
            "SHAPE_SUB": self.shape_index_subd(subdivision),
            "CORE_AREA": self.core_area(subdivision),
        }

        # ------------------- CONTENT METRICS -------------------------
        buildings_sub = self._clip_buildings(subdivision)

        # If no buildings exist in this polygon → return shape-only
        if len(buildings_sub) == 0:
            return shape_metrics

        geoms = buildings_sub.geometry.tolist()
        area_sub = subdivision.area

        # Raster for raster-based metrics (AI, CLUMPY, ENN_MN, PROX, ED)
        raster = self._rasterize_buildings(geoms, subdivision, resolution)

        # LSI (FRAGSTATS logic) based on union of building geometry
        building_union_clip = unary_union(geoms)
        if building_union_clip.is_empty or area_sub <= 0:
            LSI_value = 0.0
        else:
            P = building_union_clip.length
            A = area_sub
            LSI_value = 0.25 * P / np.sqrt(A)

        content_metrics = {
            "AI":        self.aggregation_index(raster),
            "CLUMPY":    self.clumpiness(raster),
            "ENN_MN":    self.enn_mn_raster(raster, subdivision, resolution),
            "PROX":      self.proximity_index(geoms, subdivision, resolution),
            "ED":        self.edge_density_raster(raster, subdivision, resolution),
            "LSI":       LSI_value,
            "SHAPE_MN":  float(np.mean([self.building_shape_index(g) for g in geoms])),
            "FRAC_MN":   float(np.nanmean([self.building_fractal_dimension(g) for g in geoms])),
        }

        # Merge dictionaries and return
        return {**shape_metrics, **content_metrics}


# ================================================================
# COMPACTNESS INDEX BUILDER (Entropy-weighted composite)
# ================================================================
class CompactnessIndexBuilder:
    """
    Takes raw compactness metrics from CompactnessEngine
    and builds a normalized, entropy-weighted compactness index.

    Steps:
    1. Re-orient metrics so that larger = more compact
    2. Normalize metrics (min-max)
    3. Compute entropy weights
    4. Compute weighted composite index
    """

    def __init__(self, df, metrics_main, metrics_optional):
        """
        df: DataFrame containing raw metrics per subdivision.
        metrics_main: list of core SUM metrics (AI, CLUMPY, PROX, ENN_MN, ED)
        metrics_optional: list of secondary metrics (SHAPE_MN, FRAC_MN)
        """

        self.df = df.copy()
        self.metrics_main = metrics_main
        self.metrics_optional = metrics_optional

        # full list of metrics used in the index
        self.metrics_all = metrics_main + metrics_optional

    # ------------------------------------------------------------
    # 1. FIX METRIC DIRECTION (higher = more compact)
    # ------------------------------------------------------------
    def reorient_metrics(self):
        """
        Re-orient metrics where smaller = better:
        ENN_MN, ED (dispersion/fragmentation)
        """
        # invert ENN_MN (nearest neighbor distance)
        if "ENN_MN" in self.df.columns:
            self.df["ENN_INV"] = 1 / (self.df["ENN_MN"] + 1e-6)

        # invert ED (edge density)
        if "ED" in self.df.columns:
            self.df["ED_INV"] = 1 / (self.df["ED"] + 1e-6)

        # replace original metrics with inverted metrics in the list
        if "ENN_MN" in self.metrics_all:
            self.metrics_all.remove("ENN_MN")
            self.metrics_all.append("ENN_INV")

        if "ED" in self.metrics_all:
            self.metrics_all.remove("ED")
            self.metrics_all.append("ED_INV")

        return self.df

    # ------------------------------------------------------------
    # 2. NORMALIZATION (one time only)
    # ------------------------------------------------------------
    def normalize(self):
        """
        Apply min-max normalization to metrics
        """
        for col in self.metrics_all:
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            self.df[col + "_norm"] = (self.df[col] - col_min) / (col_max - col_min + 1e-9)

        # store normalized column names
        self.norm_cols = [c + "_norm" for c in self.metrics_all]

        return self.df

    # ------------------------------------------------------------
    # 3. ENTROPY WEIGHTING
    # ------------------------------------------------------------
    def compute_entropy_weights(self):
        """
        Compute entropy weights from normalized metrics
        """
        X = self.df[self.norm_cols].values
        X = np.clip(X, 1e-12, 1)  # avoid log(0)

        # Step 1: Compute proportions p_ij
        P = X / X.sum(axis=0)

        # Step 2: Compute entropy for each metric
        n = X.shape[0]
        k = 1 / np.log(n)
        entropy = -k * (P * np.log(P)).sum(axis=0)

        # Step 3: Compute diversification degree (1 - entropy)
        d = 1 - entropy

        # Step 4: Normalize d to sum to 1
        weights = d / d.sum()

        self.weights = dict(zip(self.norm_cols, weights))

        return self.weights

    # ------------------------------------------------------------
    # 4. COMPUTE FINAL COMPOSITE INDEX
    # ------------------------------------------------------------
    def compute_index(self):
        """
        Compute weighted sum of normalized metrics.
        """
        self.df["SUM_COMPACTNESS"] = 0.0

        for col_norm, w in self.weights.items():
            self.df["SUM_COMPACTNESS"] += self.df[col_norm] * w

        return self.df[["SUM_COMPACTNESS"] + self.norm_cols]