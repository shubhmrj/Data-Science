from bs4 import BeautifulSoup
import requests

url = "https://www.sih.gov.in/sih2025PS"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

# check if table exists
table = soup.find("table", {"id": "dataTablePS"})
print(table)   # tbody will still be empty (because of JavaScript)
