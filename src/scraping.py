import os
import requests
import zipfile
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# -------------------------------
# CONFIGURATION
# -------------------------------
CHROME_VERSION = "129.0.6668.100"
CHROME_BASE = "https://storage.googleapis.com/chrome-for-testing-public"
CHROME_URL = f"{CHROME_BASE}/{CHROME_VERSION}/linux64/chrome-linux64.zip"
DRIVER_URL = f"{CHROME_BASE}/{CHROME_VERSION}/linux64/chromedriver-linux64.zip"

CHROME_DIR = "/tmp/chrome"
DRIVER_DIR = "/tmp/chromedriver"

os.makedirs(CHROME_DIR, exist_ok=True)
os.makedirs(DRIVER_DIR, exist_ok=True)

# -------------------------------
# DOWNLOAD & EXTRACT CHROME
# -------------------------------
if not os.path.exists(os.path.join(CHROME_DIR, "chrome-linux64", "chrome")):
    print("⬇️ Downloading Chrome...")
    r = requests.get(CHROME_URL)
    chrome_zip = os.path.join(CHROME_DIR, "chrome.zip")
    with open(chrome_zip, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(chrome_zip, "r") as zip_ref:
        zip_ref.extractall(CHROME_DIR)
    print("✅ Chrome extracted")

# -------------------------------
# DOWNLOAD & EXTRACT CHROMEDRIVER
# -------------------------------
if not os.path.exists(os.path.join(DRIVER_DIR, "chromedriver-linux64", "chromedriver")):
    print("⬇️ Downloading ChromeDriver...")
    r = requests.get(DRIVER_URL)
    driver_zip = os.path.join(DRIVER_DIR, "chromedriver.zip")
    with open(driver_zip, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(driver_zip, "r") as zip_ref:
        zip_ref.extractall(DRIVER_DIR)
    print("✅ ChromeDriver extracted")

# -------------------------------
# SETUP SELENIUM
# -------------------------------
chrome_binary = os.path.join(CHROME_DIR, "chrome-linux64", "chrome")
chromedriver_binary = os.path.join(DRIVER_DIR, "chromedriver-linux64", "chromedriver")
os.chmod(chromedriver_binary, 0o755)

options = Options()
options.binary_location = chrome_binary
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-gpu")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--remote-debugging-port=9222")
options.add_argument("--disable-software-rasterizer")
options.add_argument("--window-size=1920,1080")



options = Options()
options.add_argument('--headless')  # Run in headless mode
options.add_argument('--no-sandbox')  # Disable the sandbox (important in containers)
options.add_argument('--disable-dev-shm-usage')  # Disable /dev/shm usage (common in Docker)
options.add_argument('--disable-gpu')  # Disable GPU (optional)
options.add_argument('--remote-debugging-port=9222')  # Enables remote debugging

service = Service('/path/to/chromedriver')
driver = webdriver.Chrome(service=service, options=options)


service = Service(chromedriver_binary)
driver = webdriver.Chrome(service=service, options=options)

# -------------------------------
# TEST
# -------------------------------
driver.get("https://www.google.com")
print("✅ Chrome launched successfully!")
print("Page title:", driver.title)

driver.quit()
