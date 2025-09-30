import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

async def main():
    url = "https://www.sih.gov.in/sih2025PS"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Wait for table to be rendered
        await page.wait_for_selector("#dataTablePS tbody tr")

        # Get rendered HTML
        html = await page.content()
        await browser.close()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("#dataTablePS tbody tr")

    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if cols:
            ps_title = cols[1]        # adjust index based on actual table
            submissions = int(cols[-1]) if cols[-1].isdigit() else 0
            if submissions < 50:
                print(ps_title, submissions)

asyncio.run(main())
