# scrapers/gsmarena_scraper.py
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def scrape_gsmarena(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        html = page.content()
        browser.close()

    soup = BeautifulSoup(html, "lxml")

    def get_spec(label):
        row = soup.find("td", string=label)
        return row.find_next_sibling("td").text if row else None

    return {
        "display_type": get_spec("Type"),
        "battery_mah": get_spec("Capacity"),
        "charging_watt": get_spec("Charging"),
        "main_camera_mp": get_spec("Main Camera"),
    }
