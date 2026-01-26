# Import necessary libraries for spatial analysis, clustering, and visualization
import fiona, folium, shapely, geopandas as gpd, pandas as pd, numpy as np, osmnx as ox, matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from esda.moran import Moran_Local
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPoint, shape
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
from rasterio.features import shapes
from affine import Affine


#1. Calling historical tax data


# Read Mecklenburg County boundary shapefile and dissolve into a single geometry
meck_bo = gpd.read_file("../../../Data/Original_dataset/Archive/mecklenburgcounty_boundary/MecklenburgCounty_Boundary.shp")
county_geom = meck_bo.unary_union  # dissolve to a single polygon


year = 2010 # Define year

dataset_path = f"../../../Data/Histoical_tax_data/main_files/tax_{year}.shp" # Path to geodatabase with historical tax data

layers = fiona.listlayers(os.path.dirname(dataset_path)); print(layers) # List layers available in the geodatabase

tax_file = gpd.read_file(dataset_path) # Load tax parcel data for the given year

tax_file = tax_file[~tax_file.geometry.is_empty & tax_file.geometry.notnull()] # Drop records with empty or null geometries for data integrity


#2. Meta configurations

county_wgs = meck_bo.to_crs(epsg=4326)

tax_file.columns = tax_file.columns.str.lower() #Make column names lowercase

tax_file['lv_acre'] = tax_file['landvalue'] / tax_file.geometry.area / 43560 #Calculate landvalue column

residential_types = [  # List of residential building types to exclude from analysis (optional)
    'RES', 'TOWNHOUSE', 'DUP-TRIPLEX', 'CONDO', 'COMM CONDO', 'CONDO-HI', 'MFD HOME-DW',
    'APT-GDN', 'APT-TOWNHSE', 'APT-HIRISE', 'MFD HOME-SW', 'PATIO HOME', 'GROUP HOME'
]

tax_file = tax_file[ # Filter out residential parcels and any records with blank or null building descriptions
    (~tax_file['descbuildi'].isin(residential_types)) &  # exclude residential types
    (tax_file['descbuildi'].str.strip() != '') &          # exclude empty strings
    (tax_file['descbuildi'].notnull())                     # exclude null values
]

tax_file['centroid'] = tax_file.geometry.centroid # Create a new 'centroid' column with Point geometries

tax_file['x'] = tax_file.geometry.centroid.x; tax_file['y'] = tax_file.geometry.centroid.y # Add separate columns for centroid x and y coordinates

coords = tax_file[['x', 'y']].values # Extract XY coordinates from the geometries for clustering

elevation_values = tax_file["lv_acre"].values # Extract land value per acre attribute (used as a weight/feature)

w = DistanceBand.from_dataframe(tax_file, threshold=500, binary=False, alpha=-1, silence_warnings=True) # Create spatial weights matrix with inverse distance weighting within 500 meters threshold
w.transform = 'r'  # row-standardize weights


#3. Performing HAC using Spatial DBSCAN


#DBSCAN config
eps = 0.5
min_samples = 5

#Running DBSCAN
features = tax_file[['x', 'y', 'lv_acre']].values
scaled_features = StandardScaler().fit_transform(features)
db = DBSCAN(eps = eps, min_samples = min_samples).fit(scaled_features)
tax_file['cluster'] = db.labels_

tax_file['value_group'] = pd.qcut( #QUANTILE GROUPING
    tax_file['lv_acre'],
    q=10,
    labels=False,
    duplicates='drop'
)

agg_tax_file = tax_file.dissolve(by='value_group', as_index=False) # Aggregate parcels by value group


#Visualization

agg_tax_file = agg_tax_file[['value_group', 'geometry']] # Keep only necessary columns and convert any datetime to string
for col in agg_tax_file.columns:
    if pd.api.types.is_datetime64_any_dtype(agg_tax_file[col]):
        agg_tax_file[col] = agg_tax_file[col].astype(str)

agg_tax_wgs = agg_tax_file.to_crs(epsg=4326) # Reproject to WGS84 for Folium

max_val_group = agg_tax_wgs[agg_tax_wgs['value_group'] == agg_tax_wgs['value_group'].max()] # Calculate max value_group for colormap scaling

center = agg_tax_wgs.geometry.centroid.unary_union.centroid # Calculate center of map using centroid of all parcels

m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles='cartodbpositron')

folium.GeoJson( # Add county boundary to map
    county_wgs,
    name="County Boundary",
    style_function=lambda x: {"fillColor": "#00000000", "color": "black", "weight": 2},
).add_to(m)

folium.GeoJson( # Add GeoJson layer with style and tooltip
    max_val_group,
        style_function=lambda feature: {
        "color": "red",
        "weight": 0.1,
        "fillOpacity": 0.5},
).add_to(m)

m.save(f"../../../output/map/HAC/DBSCAN_last_decile_{year}.html") # Save to HTML file



#4. Performing HAC using LISA

y = tax_file['lv_acre'] # Extract dependent variable

lisa = Moran_Local(y, w) # Compute Local Moran's I

tax_file['Isig'] = lisa.p_sim < 0.05;  tax_file['Ilocal'] = lisa.Is # Add results to GeoDataFrame

sig = lisa.p_sim < 0.05;  quad = lisa.q # Classification

tax_file['cluster'] = 'Non-significant'
tax_file.loc[sig & (quad == 1), 'cluster'] = 'High-High'
tax_file.loc[sig & (quad == 2), 'cluster'] = 'Low-High'
tax_file.loc[sig & (quad == 3), 'cluster'] = 'Low-Low'
tax_file.loc[sig & (quad == 4), 'cluster'] = 'High-Low'


#Visualization

tax_file_wgs = tax_file.to_crs(epsg=4326)

high_high_group = tax_file_wgs[tax_file_wgs['cluster'] == 'High-High']; high_high_group = high_high_group[["cluster", "geometry"]] # Filtering high_high_value_group

for col in high_high_group.columns: #convert any datetime to string
    if pd.api.types.is_datetime64_any_dtype(high_high_group[col]):
        high_high_group[col] = high_high_group[col].astype(str)

center = tax_file_wgs.geometry.centroid.unary_union.centroid # Map center

m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron") # Create Folium map

folium.GeoJson( # Add county boundary to map
    county_wgs,
    name="County Boundary",
    style_function=lambda x: {"fillColor": "#00000000", "color": "black", "weight": 2},
).add_to(m)

folium.GeoJson( # Add polygons to map
    high_high_group,
        style_function=lambda feature: {
        "color": "red",
        "weight": 0.1,
        "fillOpacity": 0.5}
).add_to(m)

m.save(f"../../../output/map/HAC/lisa_high_high_{year}.html") # Save map



#5. Performing HAC using Natual Cities Approach


tri = Delaunay(coords) # Compute Delaunay triangulation from parcel coordinates

triangles = coords[tri.simplices] # Extract triangle vertices coordinates

triangle_polygons = []; kept_triangle_indices = [];  filtered_triangle_values = [] # Initialize lists to store valid triangle polygons and their average values


# Iterate over each triangle simplex
for i, simplex in enumerate(tri.simplices):
    pts = coords[simplex]
    poly = Polygon(pts)  # create polygon from triangle vertices

    # Keep polygons valid and located within county boundary
    if poly.is_valid and poly.centroid.within(county_geom):
        triangle_polygons.append(poly)
        kept_triangle_indices.append(i)

        # Calculate average land value for triangle vertices
        avg_val = np.mean(elevation_values[simplex])
        filtered_triangle_values.append(avg_val)

filtered_triangle_values = np.array(filtered_triangle_values) # Convert filtered values to numpy array

selected_mask = filtered_triangle_values > filtered_triangle_values.mean() # Select triangles with average land value above the mean

# Create GeoDataFrame of selected polygons (natural cities)
natural_city_polys = [triangle_polygons[i] for i, keep in enumerate(selected_mask) if keep]
natural_cities_gdf = gpd.GeoDataFrame(geometry=natural_city_polys, crs=tax_file.crs)

# Prepare data for web visualization by converting to WGS84 projection
natural_cities_wgs = natural_cities_gdf.to_crs(epsg=4326)

center = county_wgs.geometry.centroid # Get centroid of county for map center

m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron") # Create Folium map centered on county centroid

# Add county boundary to map (transparent fill, black border)
folium.GeoJson(
    county_wgs,
    name="County Boundary",
    style_function=lambda x: {"fillColor": "#00000000", "color": "black", "weight": 2},
).add_to(m)

# Add natural cities polygons with red fill to map
folium.GeoJson(
    natural_cities_wgs,
        style_function=lambda feature: {
        "color": "red",
        "weight": 0.1,
        "fillOpacity": 0.5}
).add_to(m)

# Save the map as HTML file
m.save(f"../../../output/map/HAC/natural_cities_map_{year}.html")



#6. Performing HAC using natural cities apporach for street node


# Download street network graph for Mecklenburg County
place_name = "Mecklenburg County, North Carolina, USA"
G = ox.graph_from_place(place_name, network_type='drive')

# Extract nodes from the graph as GeoDataFrame
nodes = ox.graph_to_gdfs(G, edges=False)

# Project nodes to metric coordinate system (EPSG:3857) for clustering
nodes_proj = nodes.to_crs(epsg=3857)

# Extract coordinates of nodes as array for clustering
coords_nodes = np.column_stack((nodes_proj.geometry.x, nodes_proj.geometry.y))

# Apply DBSCAN clustering on nodes with 400 meters epsilon, min_samples=1
db = DBSCAN(eps=400, min_samples=5).fit(coords_nodes)

# Assign cluster labels to nodes
nodes_proj['cluster'] = db.labels_

# Generate convex hull polygons for clusters with more than 3 nodes
clusters = []
for cluster_id in nodes_proj['cluster'].unique():
    group = nodes_proj[nodes_proj['cluster'] == cluster_id]
    if len(group) > 3:
        polygon = MultiPoint(group.geometry.tolist()).convex_hull
        clusters.append({'cluster': cluster_id, 'geometry': polygon})

# Create GeoDataFrame of clusters representing natural cities
natural_cities = gpd.GeoDataFrame(clusters, crs=nodes_proj.crs)

# Reproject to WGS84 for web visualization with Folium
natural_cities_latlon = natural_cities.to_crs(epsg=4326)

# Center map on average centroid of clusters
center = [natural_cities_latlon.geometry.centroid.y.mean(), natural_cities_latlon.geometry.centroid.x.mean()]

# Create Folium map centered on clusters
m = folium.Map(location=center, zoom_start=11, tiles='cartodbpositron')

# Add county boundary to map (transparent fill, black border)
folium.GeoJson(
    county_wgs,
    name="County Boundary",
    style_function=lambda x: {"fillColor": "#00000000", "color": "black", "weight": 2},
).add_to(m)

# Add each cluster polygon to the map with blue fill
for _, row in natural_cities_latlon.iterrows():
    sim_geo = folium.GeoJson(
        row['geometry'],
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'black',
            'weight': 0.1,
            'fillOpacity': 0.5
        }
    )
    sim_geo.add_to(m)

# Save the map to an HTML file
m.save(f"../../../output/map/HAC/natural_cities_map_street_nodes.html")



#7. Performing HAC using Kernel Density

#KDE config
bw_method = 0.5

xmin, ymin, xmax, ymax = tax_file.total_bounds # Extract bounding box of tax parcels to define KDE grid extent

xres = yres = 200 # Define resolution of KDE grid (number of cells in x and y direction)

xgrid = np.linspace(xmin, xmax, xres); ygrid = np.linspace(ymin, ymax, yres) # Create equally spaced grid points in x and y dimensions

xv, yv = np.meshgrid(xgrid, ygrid) # Create 2D meshgrid of coordinates

grid_coords = np.vstack([xv.ravel(), yv.ravel()]) # Stack grid coordinates into shape (2, N) for KDE evaluation

kde = gaussian_kde(coords.T, weights=elevation_values, bw_method=bw_method) # Compute weighted KDE using parcel coordinates and land value as weights

z = kde(grid_coords).reshape((yres, xres))  # Evaluate KDE values on grid points and reshape to 2D grid

threshold = np.percentile(z, 90); mask = z >= threshold # Threshold KDE to top 10% (90th percentile) to highlight dense clusters

xcell = (xmax - xmin) / xres; ycell = (ymax - ymin) / yres # Calculate cell size in x and y directions

transform = Affine.translation(xmin, ymin) * Affine.scale(xcell, ycell) # Create affine transform mapping grid cells to spatial coordinates

shapes_gen = shapes(mask.astype(np.uint8), transform=transform) # Extract polygons representing contiguous clusters above threshold from raster mask

polygons = [shape(geom) for geom, val in shapes_gen if val == 1] # Convert extracted shapes to shapely polygons


kde_decile_gdf = gpd.GeoDataFrame(geometry=polygons, crs=tax_file.crs).dissolve() # Build GeoDataFrame from polygons and dissolve to merge contiguous polygons



# Visualization

kde_decile_wgs = kde_decile_gdf.to_crs(epsg=4326)

center = kde_decile_wgs.geometry.centroid.iloc[0] # Get centroid of cluster polygons for map center

m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron") # Create Folium map centered on KDE clusters

folium.GeoJson( # Add KDE top decile polygons with purple fill to the map
    kde_decile_gdf,
        style_function=lambda feature: {
        "color": "red",
        "weight": 0.1,
        "fillOpacity": 0.5}
).add_to(m)

m.save(f"../../../output/map/HAC/kde_nonres_last_decile_{year}.html") # Save map as HTML file

