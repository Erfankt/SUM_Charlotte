import os
import geopandas as gpd
import folium
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from folium import plugins


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

YEAR = 2024
LANDUSE_CODE_COLUMN = "lusecode"

# MUST use {year} — NOT f-string — because .format(year=year) is used below
TAX_PATH_TEMPLATE = "../../../../../Erfan Dissertation/Data/Histoical_tax_data/main_files/tax_{year}.shp"

BUILDING_GDB_PATH = "../../../../Data/Final_dataset/ABT/outputs_building_overlap/Final_Buildings_SemanticIntegrated.gpkg"

OUTPUT_DIR = "../../../../Data/Misc/MHP"
DEFAULT_MAP_CENTER = (35.2265, -80.8409)


# ---------------------------------------------------------
# GEOMETRY PROCESSING
# ---------------------------------------------------------

def building_metrics(geom: Polygon):
    """
    Compute building area, bounding box area, length, and width.
    Length = longer side of the minimum rotated rectangle
    Width  = shorter side of the minimum rotated rectangle
    """
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = np.array(mrr.exterior.coords[:-1])  # remove duplicate last point

        # Compute all 4 edges of the rectangle
        edges = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))

        # Sort edge lengths (lowest = width, largest = length)
        edges_sorted = np.sort(edges)

        width = float(edges_sorted[0])
        length = float(edges_sorted[-1])

        return geom.area, mrr.area, length, width  # <-- Correct order: (length, width)

    except Exception:
        return None, None, None, None


# ---------------------------------------------------------
# MH QUALIFICATION RULE
# ---------------------------------------------------------

def mh_qualification(length_ft, width_ft):
    """Supervisor-specified Manufactured Home dimensions."""
    return (8 <= width_ft <= 48) and (28 <= length_ft <= 93)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main(year: int = YEAR, landuse_col: str = LANDUSE_CODE_COLUMN):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------
    tax_path = TAX_PATH_TEMPLATE.format(year=year)
    tax_file = gpd.read_file(tax_path)

    MHS_tax = tax_file[tax_file[landuse_col].isin(["R200", "R210"])].copy()
    R210_tax = tax_file[tax_file[landuse_col] == "R210"].copy()

    # Important: specify correct GPKG layer
    buildings = gpd.read_file(
        BUILDING_GDB_PATH
    ).to_crs(tax_file.crs)


    # -----------------------------------------------------
    # Compute building metrics
    # -----------------------------------------------------
    metrics_list = buildings.geometry.map(building_metrics).tolist()
    buildings[["area", "bbox_area", "length_ft", "width_ft"]] = pd.DataFrame(
        metrics_list, index=buildings.index
    )
    buildings["area_ratio"] = buildings["area"] / buildings["bbox_area"]

    # MH classification
    buildings["mh_candidate"] = buildings.apply(
        lambda r: mh_qualification(r["length_ft"], r["width_ft"]),
        axis=1
    )


    # -----------------------------------------------------
    # Spatial join (buildings → R210 parcels)
    # -----------------------------------------------------
    if "index_right" in R210_tax.columns:
        R210_tax = R210_tax.drop(columns=["index_right"])

    joined = gpd.sjoin(buildings, R210_tax, predicate="intersects", how="inner")

    # Total buildings
    total_counts = (
        joined.groupby("index_right")
        .size()
        .rename("total_building_number")
    )

    # MH candidates
    mh_counts = (
        joined.loc[joined["mh_candidate"]]
        .groupby("index_right")
        .size()
        .rename("total_mh_candidate_number")
    )

    R210_tax["total_building_number"] = (
        R210_tax.index.map(total_counts).fillna(0).astype(int)
    )
    R210_tax["total_mh_candidate_number"] = (
        R210_tax.index.map(mh_counts).fillna(0).astype(int)
    )


    # -----------------------------------------------------
    # CSV EXPORTS
    # -----------------------------------------------------
    MHS_tax.to_csv(os.path.join(OUTPUT_DIR, f"MHS_tax_{year}.csv"), index=False)

    R210_tax[
        ["objectid_1", "geometry", "total_building_number", "total_mh_candidate_number"]
    ].to_csv(
        os.path.join(OUTPUT_DIR, f"R210_buildingcount_{year}.csv"),
        index=False,
    )

    joined[
        ["pid", "area", "bbox_area", "length_ft", "width_ft", "area_ratio", "mh_candidate"]
    ].to_csv(
        os.path.join(OUTPUT_DIR, f"R210_buildings_{year}.csv"),
        index=False,
    )


    # -----------------------------------------------------
    # FOLIUM MAP 1 — R200/R210 Parcels Map
    # -----------------------------------------------------

    MHS_tax_wgs = MHS_tax.to_crs(epsg=4326)

    # Fix datetime encoding
    for col in MHS_tax_wgs.select_dtypes(include=["datetime64[ns]", "datetimetz"]):
        MHS_tax_wgs[col] = MHS_tax_wgs[col].astype(str)

    map_mhs = folium.Map(
        location=DEFAULT_MAP_CENTER,
        zoom_start=13,
        tiles="OpenStreetMap"
    )

    def mhs_style(feature):
        code = feature["properties"][landuse_col]
        if code == "R200":
            return {"fillColor": "blue", "color": "blue", "weight": 1, "fillOpacity": 0.5}
        elif code == "R210":
            return {"fillColor": "red", "color": "red", "weight": 1, "fillOpacity": 0.5}
        else:
            return {"fillColor": "gray", "color": "gray", "weight": 1, "fillOpacity": 0.5}

    folium.GeoJson(
        MHS_tax_wgs,
        name="Mobile Home Parcels (R200/R210)",
        style_function=mhs_style,
        tooltip=folium.GeoJsonTooltip(fields=[landuse_col], aliases=["Land Use Code:"])
    ).add_to(map_mhs)

    folium.LayerControl().add_to(map_mhs)
    map_mhs.save(os.path.join(OUTPUT_DIR, f"MHS_tax_{year}.html"))


    # -----------------------------------------------------
    # FOLIUM MAP 2 — Buildings + MH candidates
    # -----------------------------------------------------

    # Fix datetime fields BEFORE JSON conversion
    for col in R210_tax.select_dtypes(include=["datetime64[ns]", "datetimetz"]):
        R210_tax[col] = R210_tax[col].astype(str)
    for col in buildings.select_dtypes(include=["datetime64[ns]", "datetimetz"]):
        buildings[col] = buildings[col].astype(str)

    R210_tax_wgs = R210_tax.to_crs(epsg=4326)
    buildings_wgs = buildings.to_crs(epsg=4326)

    buildings_in_R210 = gpd.overlay(buildings_wgs, R210_tax_wgs, how="intersection")

    map_r210_mh = folium.Map(
        location=DEFAULT_MAP_CENTER,
        zoom_start=13,
        tiles="CartoDB positron"
    )

    def parcel_style(feature):
        return {"color": "black", "weight": 1.5, "fillOpacity": 0.1}

    parcel_layer = folium.GeoJson(
        R210_tax_wgs,
        name="R210 Parcels",
        style_function=parcel_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["objectid_1", "total_building_number", "total_mh_candidate_number"],
            aliases=["Parcel ID:", "Total Buildings:", "MH Candidates:"]
        )
    )
    parcel_layer.add_to(map_r210_mh)

    def building_style(feature):
        if feature["properties"]["mh_candidate"]:
            return {"color": "green", "fillColor": "green", "weight": 1, "fillOpacity": 0.8}
        else:
            return {"color": "red", "fillColor": "red", "weight": 1, "fillOpacity": 0.6}

    folium.GeoJson(
        buildings_in_R210,
        name="Buildings",
        style_function=building_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["length_ft", "width_ft", "area_ratio", "mh_candidate"],
            aliases=["Length (ft):", "Width (ft):", "Area Ratio:", "MH Candidate:"],
        )
    ).add_to(map_r210_mh)

    plugins.Search(
        layer=parcel_layer,
        search_label="objectid_1",
        placeholder="Search Parcel ID...",
    ).add_to(map_r210_mh)

    legend_html = """
        <div style="
            position: fixed;
            bottom: 30px; left: 30px; z-index:9999;
            width: 240px; height: 110px; background-color: white;
            border: 2px solid grey; padding: 10px; opacity: 0.9;
        ">
            <b>MH Candidate Identification</b><br>
            <span style="color:green;">■</span> MH Candidate (12–21 ft width, 40–93 ft length) <br>
            <span style="color:red;">■</span> Not MH <br>
        </div>
    """
    map_r210_mh.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(map_r210_mh)
    map_r210_mh.save(os.path.join(OUTPUT_DIR, f"R210_mh_map_{year}.html"))


    # -----------------------------------------------------
    # FOLIUM MAP 3 — Metrics Visualization
    # -----------------------------------------------------

    joined_wgs = joined.to_crs(epsg=4326)[
        ["area", "bbox_area", "length_ft", "width_ft", "mh_candidate", "geometry"]
    ]

    centroid = R210_tax_wgs.geometry.union_all().centroid
    map_r210_metrics = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=13,
        tiles="CartoDB positron"
    )

    # Parcels
    folium.GeoJson(
        R210_tax_wgs,
        name="R210 Parcels",
        style_function=lambda x: {"color": "black", "weight": 1, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["total_building_number", "total_mh_candidate_number"],
            aliases=["Total Buildings:", "MH Candidates:"]
        )
    ).add_to(map_r210_metrics)

    # Buildings
    folium.GeoJson(
        joined_wgs,
        name="Buildings",
        style_function=lambda f: {"color": "blue", "fillOpacity": 0.7},
        tooltip=folium.GeoJsonTooltip(
            fields=["area", "bbox_area", "length_ft", "width_ft", "mh_candidate"],
            aliases=["Area:", "BBox Area:", "Length:", "Width:", "MH Candidate:"]
        )
    ).add_to(map_r210_metrics)

    folium.LayerControl().add_to(map_r210_metrics)
    map_r210_metrics.save(os.path.join(OUTPUT_DIR, f"R210_buildings_metrics_{year}.html"))


    print("\n=============================================")
    print("  ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("  Directory:", os.path.abspath(OUTPUT_DIR))
    print("=============================================\n")


if __name__ == "__main__":
    main()
