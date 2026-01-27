import time, re, numpy as np, pandas as pd, geopandas as gpd, osmnx as ox
from scipy.spatial import cKDTree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# ============================================
# 1️⃣ Load Data
# ============================================
ABT_4326 = gpd.read_file("../../../../Data/Final_dataset/ABT/ABT.gpkg", layer="subdivisions").to_crs(epsg=4326)
ws_file = pd.read_csv("../../../../Data/Final_dataset/ABT/WalkScore.csv")


# edge_driver_path = r"C:/Users/ekefayat/AppData/Local/Programs/Python/Python313/msedgedriver.exe"
# edge_driver_path = r"C:/edgedriver_win64/msedgedriver.exe"
# service = Service(edge_driver_path) # Create a Service object with your driver path


def start_edge():
    edge_options = Options()
    edge_options.add_argument("--inprivate")
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--no-sandbox")

    driver = webdriver.Edge(options=edge_options)  # 🚀 Selenium Manager
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


def is_driver_alive(driver):
    if driver is None:
        return False
    try:
        # Fastest, safest ping check
        driver.execute_script("return 1;")
        return True
    except WebDriverException:
        return False
    except:
        return False

# Opening the browser directed to Google Map and then it waits for 5 seconds.
driver_google = start_edge()


ABT_4326["transit_ws"] = None
ABT_4326["groceries_ws"] = None

temp_iter = 0
for idx, row in ABT_4326.iterrows():
    if pd.notnull(ws_file.loc[idx, "transit_ws"]) and pd.notnull(ws_file.loc[idx, "groceries_ws"]):
        print(f"{idx} is already calculated!")
        continue
    if not is_driver_alive(driver_google):
        driver_google = start_edge()
    if idx % 50 == 0:
        driver_google.quit()
        driver_google = start_edge()
    try:
        searchbox_google = driver_google.find_element(By.ID, "UGojuc")
        station_centorid_address = str(row.geometry.centroid.y) + " " + str(row.geometry.centroid.x)
        searchbox_google.send_keys(station_centorid_address)
        driver_google.execute_script('document.getElementsByClassName("mL3xi")[0].click()')
        time.sleep(2)

        address = driver_google.find_element(By.XPATH,"/html/body/div[1]/div[2]/div[9]/div[8]/div/div/div[1]/div[2]/div/div[1]/div/div/div[10]/div[2]/div[2]/span[2]/span").text



        driver_google.execute_script("window.open('https://www.walkscore.com/', '_blank');")
        driver_google.switch_to.window(driver_google.window_handles[-1])
        wait = WebDriverWait(driver_google, 5)

        searchbox_walkscore = WebDriverWait(driver_google, 2).until(EC.element_to_be_clickable((By.ID,"gs-street")))
        searchbox_walkscore.send_keys(address)
        time.sleep(2)
        WebDriverWait(driver_google, 2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div[2]/div/div[1]/div/form/button'))).click()
        time.sleep(7)

        #groceries score
        if pd.isna(ws_file.loc[idx, "groceries_ws"]):

            WebDriverWait(driver_google, 2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div/div/div[2]/div[4]/div[1]/button'))).click()

            try:
                elem = driver_google.find_element(By.XPATH,"/html/body/div[6]/div[2]/div/div[2]/div[2]/div[1]/div[2]/div[1]/div/div/div[2]/div")
                style_text = elem.get_attribute("style") or ""
                match = re.search(r"height:\s*([\d\.]+)%", style_text)
                if match:
                    groceries_Score = round(float(match.group(1)), 2)
                else:
                    groceries_Score = 999  # missing
                    print(f"No groceries score data available for Sub {row[0]}!")
            except:
                groceries_Score = 999  # element missing entirely
                print(f"No groceries score data available for Sub {row[0]}!")

            ws_file.loc[idx, "groceries_ws"] = groceries_Score

            time.sleep(1)
            WebDriverWait(driver_google, 2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[6]/div[1]/button'))).click()

        else:
            print(f"Groceries score is already calculated for Sub {row[0]}!")


        #transit score
        if pd.isna(ws_file.loc[idx, "transit_ws"]):
            try:
                alt_text = driver_google.find_element(By.CSS_SELECTOR, "div[data-type='transit'].block-header-badge img")
                alt_text = alt_text.get_attribute("alt")
                transit_Score = int(re.search(r"\d+", alt_text).group())
            except:
                transit_Score = 999
                print(f"No transit score data available for Sub {row[0]}!")

            ws_file.loc[idx, "transit_ws"] = transit_Score

        else:
            print(f"Transit score is already calculated for Sub {row[0]}!")


        WebDriverWait(driver_google, 2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[1]/div[2]/div[1]/a'))).click()

        driver_google.switch_to.window(driver_google.window_handles[-1])
        driver_google.close()

        driver_google.switch_to.window(driver_google.window_handles[-1])

        WebDriverWait(driver_google, 2).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[2]/div[9]/div[3]/div[1]/div[1]/div/div[1]/div[2]/button'))).click()

        print(f'Sub {row[0]} has been updated! transit_score: {ws_file.at[idx, "transit_ws"]}, groceries_score: {ws_file.at[idx, "groceries_ws"]}')

        time.sleep(2)

    except Exception as e:
        print(f"[{idx}] ❌ Error: {e}")
        ws_file[["subd_id", "transit_ws", "groceries_ws"]].to_csv("../../../../Data/Final_dataset/ABT/WalkScore.csv")
        driver_google.quit()
        time.sleep(10)
        continue

ws_file[["subd_id", "transit_ws", "groceries_ws"]].to_csv("../../../../Data/Final_dataset/ABT/WalkScore.csv")

