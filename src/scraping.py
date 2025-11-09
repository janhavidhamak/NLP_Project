
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time, random

service = Service(r"C:\WebDrivers\chromedriver.exe")
options = Options()

# Use temporary Chrome profile to avoid crashes
options.add_argument("--user-data-dir=C:/Users/Janhavi/TempChromeProfile")
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Random user-agent for safety
agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/118.0 Safari/537.36"
]
options.add_argument(f"user-agent={random.choice(agents)}")

driver = webdriver.Chrome(service=service, options=options)
reviews = []

for page in range(1, 9):  # up to 20 pages
    url = f"https://www.amazon.com/product-reviews/B0CZHT35WQ/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}"
    #url = f"https://www.amazon.in/product-reviews/B0D5BN76MK/ref=cm_cr_getr_d_paging_btm_next_{page}?pageNumber={page}&reviewerType=all_reviews"
#B0CRCMCBTJ
    driver.get(url)
    time.sleep(10)

    # Handle login prompt
    if "ap/signin" in driver.current_url:
        print("Amazon login page detected! Please log in manually once in the opened browser.")
        input("Press ENTER after logging in...")
        continue

    # Scroll multiple times to load all reviews
    for _ in range(5):
        driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
        time.sleep(random.uniform(1.5, 3))

    # Click "See more" / "Read more" to expand reviews
    see_more_buttons = driver.find_elements(By.XPATH, "//span[@data-action='columnbalancing-showfullreview']/a")
    for btn in see_more_buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(1)
        except:
            pass

    # Collect reviews
    try:
        elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "//span[@data-hook='review-body']"))
        )
    except:
        print(f"No reviews found on page {page}")
        continue

    for e in elements:
        text = e.text.strip()
        if text:
            reviews.append(text)

    print(f"Page {page} done. Total reviews collected: {len(reviews)}")

driver.quit()

# Save unique reviews
df = pd.DataFrame(reviews, columns=["Review"])
# df.drop_duplicates(inplace=True)
df.to_csv("amazon_reviews.csv", index=False, encoding="utf-8")
print(f"\n Saved {len(df)} unique reviews to amazon_reviews.csv")



