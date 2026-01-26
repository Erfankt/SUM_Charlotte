import osmnx as ox, pandas as pd, geopandas as gpd, networkx as nx, numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

print("\n==============================================================")
print(" STEP 1 — LOADING DATA")
print("==============================================================")

meck_bo = gpd.read_file(
    "../../../../Data/Original_dataset/Archive/mecklenburgcounty_boundary/MecklenburgCounty_Boundary.shp"
).to_crs(epsg=4326)
print("Loaded Mecklenburg boundary.")

ABT = gpd.read_file(
    "../../../../Data/Final_dataset/ABT/ABT.gpkg", layer="subdivisions"
).to_crs(epsg=4326)
print(f"Loaded ABT subdivisions: {len(ABT):,} records.")

transit_stations = gpd.read_file(
    "../../../../Data/Original_dataset/transit_stations_4326.shp"
).to_crs(epsg=4326)
print(f"Loaded transit stations: {len(transit_stations):,} records.")

groceries_dataset = gpd.read_file(
    "../../../../Data/Original_dataset/Archive/Groceries/Grocery_Stores/Grocery_Stores_(points).shp"
).to_crs(epsg=4326)

pharmacies_dataset = gpd.read_file(
    "../../../../Data/Original_dataset/Archive/Pharmacies/Pharmacies.shp"
).to_crs(epsg=4326)

print("\nCleaning invalid geometries...")
before_g = len(groceries_dataset)
before_p = len(pharmacies_dataset)

groceries_dataset = groceries_dataset[
    groceries_dataset.geometry.notnull() &
    groceries_dataset.geometry.is_valid &
    ~groceries_dataset.geometry.is_empty
].copy()

pharmacies_dataset = pharmacies_dataset[
    pharmacies_dataset.geometry.notnull() &
    pharmacies_dataset.geometry.is_valid &
    ~pharmacies_dataset.geometry.is_empty
].copy()

print(f"Groceries cleaned: {before_g} → {len(groceries_dataset)}")
print(f"Pharmacies cleaned: {before_p} → {len(pharmacies_dataset)}")

# ==============================================================
# 2. BUILD WALKING NETWORK
# ==============================================================

print("\n==============================================================")
print(" STEP 2 — BUILDING WALKING NETWORK")
print("==============================================================")

print("Requesting pedestrian network from OpenStreetMap...")
G_walk = ox.graph_from_polygon(meck_bo.geometry.iloc[0], network_type='walk')
print("Network downloaded!")

print("Projecting to EPSG:2264 (NC State Plane meters)...")
G_walk = ox.project_graph(G_walk, to_crs='EPSG:2264')

# Assign walking times
print("Assigning walking times to edges...")
WALK_SPEED = 1.4
for u, v, k, data in G_walk.edges(keys=True, data=True):
    data["time_min"] = data.get("length", 0) / (WALK_SPEED * 60)

print("Walking network prepared.")

# ==============================================================
# 3. COMPUTE CENTROIDS & SNAP TO NETWORK
# ==============================================================

print("\n==============================================================")
print(" STEP 3 — COMPUTING CENTROIDS & SNAPPING TO NETWORK")
print("==============================================================")

ABT["centroid"] = ABT.geometry.centroid
sub_pts = ABT.set_geometry("centroid").to_crs(G_walk.graph["crs"])
print("Centroids computed and projected.")

print("Snapping centroids to nearest walkable nodes...")
sub_pts["origin_node"] = ox.distance.nearest_nodes(
    G_walk, sub_pts.geometry.x, sub_pts.geometry.y
)
print("All centroids snapped to walk network.")

# ==============================================================
# 4. PREPARE TRANSIT STATION COORDS + KDTREE
# ==============================================================

print("\n==============================================================")
print(" STEP 4 — BUILDING KDTREE FOR TRANSIT STATIONS")
print("==============================================================")

sta_pts = transit_stations.to_crs(G_walk.graph["crs"])
station_coords = np.array(list(zip(sta_pts.geometry.x, sta_pts.geometry.y)))
tree_stations = cKDTree(station_coords)
print("KDTree for transit stations created.")

# ==============================================================
# 5. PREPARE COMBINED AMENITY DATASET
# ==============================================================

print("\n==============================================================")
print(" STEP 5 — COMBINING GROCERY + PHARMACY DATASETS")
print("==============================================================")

groceries_dataset["amenity_type"] = "Grocery"
groceries_dataset["amenity_id"]   = groceries_dataset["BWID"]

pharmacies_dataset["amenity_type"] = "Pharmacy"
pharmacies_dataset["amenity_id"]   = pharmacies_dataset["PID"]

amenities = pd.concat(
    [groceries_dataset[["geometry", "amenity_type", "amenity_id"]],
     pharmacies_dataset[["geometry", "amenity_type", "amenity_id"]]],
    ignore_index=True
).to_crs(G_walk.graph["crs"])

print(f"Total amenities combined: {len(amenities):,}")

amenity_coords = np.array(list(zip(amenities.geometry.x, amenities.geometry.y)))
tree_amenities = cKDTree(amenity_coords)
print("KDTree for amenities created.")

# ==============================================================
# 6. HELPER FUNCTIONS
# ==============================================================

print("\n==============================================================")
print(" STEP 6 — DEFINING NEAREST-NEIGHBOR FUNCTIONS")
print("==============================================================")

def get_network_nearest_station(G, origin_node, station_coords, tree, origin_xy, k=3):
    dist, nearest_idxs = tree.query(origin_xy, k=k)
    if np.isscalar(nearest_idxs):
        nearest_idxs = [nearest_idxs]

    best_time = np.inf
    best_idx, best_node = None, None

    for idx in nearest_idxs:
        sx, sy = station_coords[idx]
        try:
            node = ox.distance.nearest_nodes(G, sx, sy)
            t_min = nx.shortest_path_length(G, origin_node, node, weight="time_min")
            if t_min < best_time:
                best_time, best_idx, best_node = t_min, idx, node
        except:
            continue

    return best_idx, best_node, (np.nan if best_time == np.inf else best_time)


def get_network_nearest_amenity(G, origin_node, coords, tree, origin_xy, k=3):
    dist, nearest_idxs = tree.query(origin_xy, k=k)
    if np.isscalar(nearest_idxs):
        nearest_idxs = [nearest_idxs]

    best_time = np.inf
    best_idx, best_node = None, None

    for idx in nearest_idxs:
        ax, ay = coords[idx]
        try:
            node = ox.distance.nearest_nodes(G, ax, ay)
            t_min = nx.shortest_path_length(G, origin_node, node, weight="time_min")
            if t_min < best_time:
                best_time, best_idx, best_node = t_min, idx, node
        except:
            continue

    return best_idx, best_node, (np.nan if best_time == np.inf else best_time)

print("Helper functions initialized.")

# ==============================================================
# 7. COMPUTE ACCESSIBILITY
# ==============================================================

print("\n==============================================================")
print(" STEP 7 — COMPUTING ACCESSIBILITY (TRANSIT + AMENITIES)")
print("==============================================================")

sub_pts["nearest_station_idx"] = np.nan
sub_pts["nearest_station_node"] = np.nan
sub_pts["walk_time_min_osm"] = np.nan

sub_pts["nearest_amenity_idx"] = np.nan
sub_pts["nearest_amenity_node"] = np.nan
sub_pts["walk_time_min_amenity"] = np.nan
sub_pts["amenity_type"] = None
sub_pts["amenity_id"] = None

print(f"Processing {len(sub_pts):,} subdivisions...\n")

for i, (idx, row) in enumerate(sub_pts.iterrows()):
    if i % 100 == 0:
        print(f" → Progress: {i:,}/{len(sub_pts):,} subdivisions processed...")

    origin_node = row["origin_node"]
    origin_xy = (row["centroid"].x, row["centroid"].y)

    # Transit
    s_idx, s_node, s_time = get_network_nearest_station(
        G_walk, origin_node, station_coords, tree_stations, origin_xy
    )
    sub_pts.loc[idx, "nearest_station_idx"] = s_idx
    sub_pts.loc[idx, "nearest_station_node"] = s_node
    sub_pts.loc[idx, "walk_time_min_osm"] = s_time

    # Amenities
    a_idx, a_node, a_time = get_network_nearest_amenity(
        G_walk, origin_node, amenity_coords, tree_amenities, origin_xy
    )
    sub_pts.loc[idx, "nearest_amenity_idx"] = a_idx
    sub_pts.loc[idx, "nearest_amenity_node"] = a_node
    sub_pts.loc[idx, "walk_time_min_amenity"] = a_time

    if a_idx is not None and not np.isnan(a_idx):
        sub_pts.loc[idx, "amenity_type"] = amenities.iloc[int(a_idx)]["amenity_type"]
        sub_pts.loc[idx, "amenity_id"] = amenities.iloc[int(a_idx)]["amenity_id"]

print("\nAccessibility computation complete.")

# ==============================================================
# 8. ATTACH TRANSIT StopID
# ==============================================================

print("\n==============================================================")
print(" STEP 8 — ATTACHING STOP IDs")
print("==============================================================")

sub_pts["StopID"] = sub_pts["nearest_station_idx"].apply(
    lambda i: transit_stations.iloc[int(i)]["StopID"] if pd.notna(i) else None
)

print("Stop IDs joined.")

# ==============================================================
# 9. EXPORT FINAL RESULTS
# ==============================================================

print("\n==============================================================")
print(" STEP 9 — EXPORTING RESULTS")
print("==============================================================")

out = sub_pts[[
    "subd_id",
    "StopID",
    "walk_time_min_osm",
    "amenity_type",
    "amenity_id",
    "walk_time_min_amenity"
]]

out.to_csv("../../../../Data/Final_dataset/ABT/accessibility_osm.csv", index=False)

print("\nCSV exported → accessibility_osm.csv")

# ==============================================================
# 10. SUMMARY STATS
# ==============================================================

print("\n==============================================================")
print(" SUMMARY")
print("==============================================================")

print("Transit walking time:")
print(out["walk_time_min_osm"].describe())

print("\nAmenity walking time:")
print(out["walk_time_min_amenity"].describe())

print("\nDONE.\n")