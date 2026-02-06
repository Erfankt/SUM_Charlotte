import time, re, pandas as pd, geopandas as gpd
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains



# ============================================
# 1️⃣ Load Data
# ============================================
ws_file = pd.read_csv("../../../../MHP/MH-new data-Jan 2026/unique_mh_addresses_missing_coords.csv")


def start_edge():
    edge_options = Options()
    edge_options.add_argument("--inprivate")
    edge_options.add_argument("--disable-blink-features=AutomationControlled")
    edge_options.add_argument("--disable-dev-shm-usage")
    edge_options.add_argument("--no-sandbox")

    driver = webdriver.Edge(options=edge_options)
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

temp_iter = 0
for idx, row in ws_file.iterrows():
    searchbox_google = driver_google.find_element(By.ID, "UGojuc")
    station_centorid_address = str(row[0])
    searchbox_google.send_keys(station_centorid_address)
    driver_google.execute_script('document.getElementsByClassName("mL3xi")[0].click()')
    time.sleep(6)
    driver_google.execute_script('document.getElementsByClassName("rM8v5b FpkVFf")[0].click()')
    driver_google.execute_script('document.getElementsByClassName("rM8v5b FpkVFf")[0].click()')
    driver_google.execute_script('document.getElementsByClassName("rM8v5b FpkVFf")[0].click()')

    # Rightclick on the google map driver
    action = ActionChains(driver_google)
    right_click_pointer = driver_google.find_element(By.TAG_NAME, 'canvas')
    double_click = action.context_click(right_click_pointer)
    double_click.perform()
    time.sleep(1)
    coordinate_google = driver_google.find_element(By.CLASS_NAME, "fxNQSd").text

    # getting the google coordinates
    lat_google = coordinate_google.split(",")[0]
    long_google = coordinate_google.split(",")[1]
    long_google = long_google[1:]

    ws_file.loc[idx, ws_file.columns[1]] = lat_google
    ws_file.loc[idx, ws_file.columns[2]] = long_google

    print(f"{idx} done!")

    driver_google.find_element(By.ID, "UGojuc").clear()
    driver_google.execute_script('document.getElementsByClassName("yAuNSb vF7Cdb")[0].click()')


ws_file.to_csv("../../../../MHP/MH-new data-Jan 2026/unique_mh_addresses_missing_coords.csv", index=False)

