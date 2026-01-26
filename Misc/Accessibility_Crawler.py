import time, re, numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options 




# ============================================
# 1️⃣ Load Data
# ============================================
transit_stations_4326 = gpd.read_file("../../../Data/Original_dataset/transit_stations_4326.shp").to_crs(epsg=4326)
ABT_4326 = gpd.read_file("../../../Data/Final_dataset/ABT/ABT.gpkg", layer="subdivisions").to_crs(epsg=4326)
groceries_dataset = gpd.read_file("../../../Data/Original_dataset/Archive/Groceries/Grocery_Stores/Grocery_Stores_(points).shp")
pharmacies_dataset = gpd.read_file("../../../Data/Original_dataset/Archive/Pharmacies/Pharmacies.shp")
groceries_dataset = pd.concat([groceries_dataset, pharmacies_dataset], ignore_index=True)
meck_bo = gpd.read_file("../../../Data/Original_dataset/Archive/mecklenburgcounty_boundary/MecklenburgCounty_Boundary.shp").to_crs(epsg=4326)
groceries_dataset_4326 = groceries_dataset[groceries_dataset.within(meck_bo.geometry.iloc[0])].to_crs(epsg=4326)
accessibility_final = pd.read_excel("../../../Data/Final_dataset/ABT/accessibility_analysis_final.xlsx")


already_computed_ids = accessibility_final['subd_id'].tolist()


# Compute centroids
ABT_4326["centroid"] = ABT_4326.geometry.centroid
ABT_4326["centroid_lon"] = ABT_4326.centroid.x
ABT_4326["centroid_lat"] = ABT_4326.centroid.y


# Build KDTree for APT
station_coords = np.array(list(zip(transit_stations_4326.geometry.x, transit_stations_4326.geometry.y)))
tree_stations = cKDTree(station_coords)

# Build KDTree for AUO
groceries_coords = np.array(list(zip(groceries_dataset_4326.geometry.x, groceries_dataset_4326.geometry.y)))
tree_groceries = cKDTree(groceries_coords)


ABT_4326["APT"] = None
ABT_4326["APT_station"] = None
ABT_4326["AUO"] = None
ABT_4326["AUO_Grocery"] = None


edge_driver_path = r"C:/Users/ekefayat/AppData/Local/Programs/Python/Python313/msedgedriver.exe" # Path to your msedgedriver.exe
# edge_driver_path = r"C:/edgedriver_win64/msedgedriver.exe"
service = Service(edge_driver_path) # Create a Service object with your driver path


def start_edge():
    edge_options = Options()
    edge_options.add_argument("--inprivate")  # starts Edge in InPrivate mode
    driver = webdriver.Edge(service=service, options=edge_options)
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.get("https://www.google.com/maps")
    return driver

def click_if_exists(driver, xpath, timeout=2):
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        return True
    except TimeoutException:
        # Element does not exist or is not clickable within the timeout
        return False


# Opening the browser directed to Google Map and then it waits for 5 seconds.
driver_google = start_edge()

temp_iter = 0
for idx, row in ABT_4326.iterrows():
    if row["subd_id"] not in already_computed_ids:
        try:
            #Handling APT
            rows = []
            sub_x, sub_y = row["centroid_lon"], row["centroid_lat"]
            distances, indices = tree_stations.query([sub_x, sub_y], k=3)
            nearest_station_idx = indices[np.argmin(distances)]
            nearest_station_geom = transit_stations_4326.iloc[nearest_station_idx].geometry
            nearest_station_name = transit_stations_4326.iloc[nearest_station_idx]["StopDesc"]
            searchbox_google = driver_google.find_element(By.ID, "searchboxinput")
            station_centorid_address = str(nearest_station_geom.y) + " " + str(nearest_station_geom.x)
            searchbox_google.send_keys(station_centorid_address)
            driver_google.execute_script('document.getElementsByClassName("mL3xi")[0].click()')
            time.sleep(2)
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[4]/div[1]/button'))).click()
            time.sleep(2)
            searchbox_origin = driver_google.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[2]/div[1]/div[1]/div[2]/div[1]/div/input")
            searchbox_origin.clear()
            origin_centorid_address = str(row['centroid_lat']) + " " + str(row['centroid_lon'])
            searchbox_origin.send_keys(origin_centorid_address)
            time.sleep(2)
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[2]/div[1]/div[1]/div[2]/button[1]'))).click()
            time.sleep(2)
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[1]/div/div/div/div[4]/button/div[1]'))).click()
            time.sleep(2)
            travel_time = driver_google.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[5]/div/div[1]/div/div[1]/div[1]").text
            # Extract hours and minutes if present
            hours = re.search(r'(\d+)\s*hr', travel_time)
            mins = re.search(r'(\d+)\s*min', travel_time)
            total_mins = (int(hours.group(1)) * 60 if hours else 0) + (int(mins.group(1)) if mins else 0) # Convert to total minutes
            travel_distance = driver_google.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[5]/div/div[1]/div/div[1]/div[2]").text
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[1]/div/button/span'))).click()
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[1]/div/div[1]/div[2]/button'))).click()

            ABT_4326.loc[idx, "APT"] = total_mins
            ABT_4326.loc[idx, "APT_station"] = nearest_station_name

            print(f"subdivision {idx} has been updated successfully: travel time: {total_mins} minutes to the nearest transit station.")


            #Handling AUO
            rows = []
            distances_groceries, indices_groceries = tree_groceries.query([sub_x, sub_y], k=3)
            nearest_groceries_idx = indices_groceries[np.argmin(distances_groceries)]
            nearest_groceries_geom = groceries_dataset_4326.iloc[nearest_groceries_idx].geometry
            nearest_groceries_name = groceries_dataset_4326.iloc[nearest_groceries_idx]["Address"]

            searchbox_google = driver_google.find_element(By.ID, "searchboxinput")
            groceries_centorid_address = str(nearest_groceries_geom.y) + " " + str(nearest_groceries_geom.x)
            searchbox_google.send_keys(groceries_centorid_address)
            driver_google.execute_script('document.getElementsByClassName("mL3xi")[0].click()')
            time.sleep(2)
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[4]/div[1]/button'))).click()
            time.sleep(2)
            searchbox_origin = driver_google.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[2]/div[1]/div[1]/div[2]/div[1]/div/input")
            searchbox_origin.clear()
            origin_centorid_address = str(row['centroid_lat']) + " " + str(row['centroid_lon'])
            searchbox_origin.send_keys(origin_centorid_address)
            time.sleep(2)
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[2]/div[1]/div[1]/div[2]/button[1]'))).click()
            time.sleep(2)
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[1]/div/div/div/div[4]/button/div[1]'))).click()
            time.sleep(2)
            travel_time = driver_google.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[5]/div/div[1]/div/div[1]/div[1]").text
            # Extract hours and minutes if present
            hours = re.search(r'(\d+)\s*hr', travel_time)
            mins = re.search(r'(\d+)\s*min', travel_time)
            total_mins = (int(hours.group(1)) * 60 if hours else 0) + (int(mins.group(1)) if mins else 0) # Convert to total minutes
            travel_distance = driver_google.find_element(By.XPATH, "/html/body/div[1]/div[3]/div[9]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[5]/div/div[1]/div/div[1]/div[2]").text
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[1]/div/button/span'))).click()
            button = WebDriverWait(driver_google,2).until(EC.element_to_be_clickable((By.XPATH,'/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[1]/div/div[1]/div[2]/button'))).click()

            ABT_4326.loc[idx, "AUO"] = total_mins
            ABT_4326.loc[idx, "AUO_Grocery"] = nearest_groceries_name

            print(f"subdivision {idx} has been updated successfully: travel time: {total_mins} minutes to the nearest groceries.")


        except Exception as e:
            print(f"[{idx}] ❌ Error: {e}")
            ABT_4326.loc[idx, "APT"] = None
            ABT_4326.loc[idx, "APT_station"] = None
            ABT_4326.loc[idx, "AUO"] = None
            ABT_4326.loc[idx, "AUO_Grocery0"] = None
            ABT_4326[["subd_id", "APT", "APT_station", "AUO", "AUO_Grocery"]].to_csv("../../../Data/Final_dataset/ABT/accessibility_analysis.csv")
            click_if_exists(driver_google, '/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[2]/div/div[1]/div/button/span')
            click_if_exists(driver_google, '/html/body/div[1]/div[3]/div[9]/div[3]/div[1]/div[1]/div/div[1]/div[2]/button')
            continue

        temp_iter = temp_iter + 1

        if temp_iter % 70 == 0:
            driver_google.close()
            driver_google = start_edge()

        print(temp_iter)
    else:
        print(f" subdivision {idx} is already computed!")


