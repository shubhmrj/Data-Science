from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re

url = "https://www.sih.gov.in/sih2025PS"

# Start Chrome (make sure you have ChromeDriver installed)
driver = webdriver.Chrome()
driver.get(url)

# Wait for table to load using explicit waits
wait = WebDriverWait(driver, 20)
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#dataTablePS tbody tr")))

# Try to show more rows per page if the length selector exists
try:
    length_select_el = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#dataTablePS_length select")))
    Select(length_select_el).select_by_value("100")
    # wait for table to refresh
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#dataTablePS tbody tr")))
except Exception:
    pass

def extract_indices(soup):
    """Map header names to indices for title and submissions."""
    headers = [th.get_text(strip=True).lower() for th in soup.select("#dataTablePS thead th")]
    title_idx = None
    sub_idx = None
    for i, h in enumerate(headers):
        if title_idx is None and ("title" in h or "problem" in h):
            title_idx = i
        if sub_idx is None and ("submission" in h or "submissions" in h):
            sub_idx = i
    # fallbacks if not found
    if title_idx is None:
        title_idx = 1 if len(headers) > 1 else 0
    if sub_idx is None:
        sub_idx = len(headers) - 1 if headers else -1
    return title_idx, sub_idx

def clean_int(text):
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else 0

def parse_current_page(html, under_50):
    soup = BeautifulSoup(html, "html.parser")
    title_idx, sub_idx = extract_indices(soup)
    rows = soup.select("#dataTablePS tbody tr")
    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if not cols:
            continue
        # guard for index range
        if title_idx >= len(cols) or sub_idx >= len(cols) or sub_idx < 0:
            continue
        ps_title = cols[title_idx]
        submissions = clean_int(cols[sub_idx])
        if submissions < 50:
            under_50.append((ps_title, submissions))

under_50 = []

# Iterate through all pages using the DataTables next button
while True:
    # Parse this page
    html = driver.page_source
    parse_current_page(html, under_50)

    # Find the next button and check if it's disabled
    try:
        next_btn = driver.find_element(By.CSS_SELECTOR, "#dataTablePS_next")
        classes = next_btn.get_attribute("class") or ""
        if "disabled" in classes:
            break
        # capture first row text to detect page change
        first_row_text = driver.find_element(By.CSS_SELECTOR, "#dataTablePS tbody tr").text
        next_btn.click()
        # wait until the first row changes (page advanced)
        wait.until(lambda d: d.find_element(By.CSS_SELECTOR, "#dataTablePS tbody tr").text != first_row_text)
    except Exception:
        # If anything goes wrong with pagination, stop to avoid infinite loops
        break

driver.quit()

# Print results
for title, subs in under_50:
    print(title, subs)
